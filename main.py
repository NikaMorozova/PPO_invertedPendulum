import argparse
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from test import test
from train import train


def f_and_t(s):
    if s.lower() in ('true', 'yes', 'y', '1'):
        return True
    elif s.lower() in ('false', 'no','n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def main(args=None):
    if args.run.lower() == "train":
        train(
            upswing=args.upswing,
            target=args.target,
            num_epochs=args.num_epochs,
            horizon=args.horizon
        )
    if args.run.lower() == "test":
        test(
            upswing=args.upswing,
            target=args.target,
            mass=args.mass,
            actor_weights=args.actor_weights,
            critic_weights=args.critic_weights
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--run',dest='run', type=str, default='train', help='Choose Train or Test mode')
    parser.add_argument('--upswing', dest='upswing', type=f_and_t, default=False, help='Choose whether you want to solve upswing task')
    parser.add_argument('--target', dest='target', type=f_and_t, default=False, help='Choose whether you want to solve target task')
    parser.add_argument('--mass', dest='mass', type=float, default=None, help='Pendulum mass value')
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=500)
    parser.add_argument('--horizon', dest='horizon', type=int, default=2048)
    parser.add_argument('--actor_weights', dest='actor_weights', type=str, default=None)
    parser.add_argument('--critic_weights', dest='critic_weights', type=str, default=None)
    args = parser.parse_args()

    timestamp = str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

    log_dir = os.path.join("experiments", args.run, timestamp).replace("\\", "/")

    writer = SummaryWriter(log_dir=log_dir)
    main(args)
