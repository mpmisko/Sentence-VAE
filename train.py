import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from ptb import PTB
from utils import to_var, idx2word, expierment_name
from model import SentenceVAE, loss_fn, Discriminator

def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    splits = ['train', 'valid'] + (['test'] if args.test else [])

    datasets = OrderedDict()
    for split in splits:
        datasets[split] = PTB(
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ
        )

    params = dict(
        vocab_size=datasets['train'].vocab_size,
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        unk_idx=datasets['train'].unk_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
    )

    NLL = torch.nn.NLLLoss(ignore_index=datasets['train'].pad_idx, reduction='none')
    g1 = SentenceVAE(**params)
    g2 = SentenceVAE(**params)
    d = Discriminator(**params)
    opt1 = torch.optim.Adam(g1.parameters(), lr=args.learning_rate)
    opt2 = torch.optim.Adam(g2.parameters(), lr=args.learning_rate)
    opt_d = torch.optim.Adam(d.parameters(), lr=args.learning_rate)
    is_d = True
    g_active, g_passive = g1, g2
    opt_active, opt_passive = opt1, opt2

    print("Generator 1:", g1)
    print("Generator 2:", g2)
    print("Discriminator:", d)

    if torch.cuda.is_available():
        g1 = g1.cuda()
        g2 = g2.cuda()
        d = d.cuda()

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, expierment_name(args, ts)))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    for epoch in range(args.epochs):

        for split in splits:

            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split=='train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                g1.train()
                g2.train()
                d.train()
            else:
                g1.eval()
                g2.eval()
                d.eval()

            for iteration, batch in enumerate(data_loader):

                batch_size = batch['input'].size(0)

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Generate fake sequences from passive
                x_fake, z, passive_nll = g_passive.inference(NLL, args, n=batch_size, ts=batch['target'], lens=batch['length'])
                fake_lengths = torch.ones([batch_size]).long() * args.max_sequence_length

                # Compute log likelihood from the active generator
                logp, mean, logv, z = g_active(x_fake, fake_lengths)
                NLL_loss, KL_loss, KL_weight = loss_fn(NLL, logp, batch['target'],
                    batch['length'], mean, logv, args.anneal_function, step, args.k, args.x0)
                active_nll = NLL_loss + KL_weight * KL_loss
                active_real_nll = g_active.get_seq_likelihood(batch['target'], batch['length'], NLL, args)
                
                d_real = torch.log(1 - d(batch['input'], batch['length']) + 1e-3)
                d_fake = torch.log(1 - d(x_fake, fake_lengths) + 1e-3)
                g_objective = torch.mean(d_fake - active_nll + passive_nll)
                g_objective += torch.mean(d_real - active_real_nll + passive_real_nll)
                g_objective /= 2

                d_real = d(batch['input'], batch['length'])
                d_objective = -torch.mean(d_fake) - torch.mean(torch.log(d_real + 1e-3))

                # backward + optimization
                if split == 'train':
                    if is_d:
                        opt_d.zero_grad()
                        d_objective.backward()
                        opt_d.step()
                    else:
                        opt_active.zero_grad()
                        g_objective.backward()
                        opt_active.step()
                    step += 1

                if is_d and iteration % args.switch == 0:
                    is_d = False
                    g_active, g_passive = g_passive, g_active
                    opt_active, opt_passive = opt_passive, opt_active
                elif iteration % args.switch == 0:
                    is_d = True

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch %04d/%i, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f, GEN %9.4f, DISC %9.4f"
                          % (split.upper(), iteration, len(data_loader)-1, torch.mean(NLL_loss).item(),
                          torch.mean(KL_loss).item(), KL_weight, g_objective.item(), d_objective.item()))

                if split == 'valid':
                    if 'target_sents' not in tracker:
                        tracker['target_sents'] = list()
                    tracker['target_sents'] += idx2word(batch['target'].data, i2w=datasets['train'].get_i2w(),
                                                        pad_idx=datasets['train'].pad_idx)
                    tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)


            # save a dump of all sentences and the encoded latent space
            if split == 'valid':
                dump = {'target_sents': tracker['target_sents'], 'z': tracker['z'].tolist()}
                if not os.path.exists(os.path.join('dumps', ts)):
                    os.makedirs('dumps/'+ts)
                with open(os.path.join('dumps/'+ts+'/valid_E%i.json' % epoch), 'w') as dump_file:
                    json.dump(dump,dump_file)

            # save checkpoint
            if split == 'train':
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % epoch)
                torch.save(g1.state_dict(), checkpoint_path)
                print("Model saved at %s" % checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-s', '--switch', type=int, default=5)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='logs')
    parser.add_argument('-bin', '--save_model_path', type=str, default='bin')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.anneal_function in ['logistic', 'linear']
    assert 0 <= args.word_dropout <= 1

    main(args)
