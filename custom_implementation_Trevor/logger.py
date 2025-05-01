from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self,args) -> None:
        if not args.evaluate:
            print("Running training")
            self.writer = SummaryWriter(f"runs/{args.run_name}")
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),)
        else:
            self.writer=None
            print("Running evaluation")

    def add_scalar(self, tag, scalar_value, step):
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()

