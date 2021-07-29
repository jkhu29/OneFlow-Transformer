import tqdm

import oneflow.experimental as flow
import oneflow.experimental.nn as nn

from eager.language_model import TransLM
from eager.transformer import Transformer
from trainer.optim_schedule import ScheduledOptim


class TransTrainer:
    """
    TransTrainer make the pretrained Transformer model
    """

    def __init__(
        self,
        trans: Transformer,
        vocab_size: int,
        train_dataloader=None,
        test_dataloader=None,
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        weight_decay: float = 0.01,
        warmup_steps=10000,
        with_cuda: bool = True,
        cuda_devices=None,
        log_freq: int = 10,
    ):
        """
        :param trans: Transformer model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for Transformer training, argument -c, --cuda should be true
        # cuda_condition = flow.cuda.is_available() and with_cuda
        cuda_condition = with_cuda
        self.device = flow.device("cuda:0" if cuda_condition else "cpu")

        # This Transformer model will be saved every epoch
        self.trans = trans.to(self.device)
        # Initialize the Transformer Language Model, with Transformer model
        self.model = BERTLM(trans, vocab_size).to(self.device)
        # self.model.load_state_dict(flow.load("output/init"))

        # # Distributed GPU training if CUDA can detect more than 1 GPU
        # if with_cuda and flow.cuda.device_count() > 1:
        #     print("Using %d GPUS for Transformer" % flow.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # # Setting the Adam optimizer with hyper-param
        self.optim = flow.optim.Adam(
            self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
        self.optim_schedule = ScheduledOptim(
            self.optim, self.trans.hidden, n_warmup_steps=warmup_steps
        )

        self.criterion = nn.NLLLoss(ignore_index=0)
        self.criterion = self.criterion.to(self.device)

        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (str_code, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}",
        )

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            # TODO

            # 3. backward and optimization only in train
            if train:
                loss.backward()
                self.optim_schedule.step_and_update_lr()
                self.optim_schedule.zero_grad()

            # next sentence prediction accuracy
            correct = (
                next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().numpy().item()
            )
            avg_loss += loss.numpy().item()
            total_correct += correct
            total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.numpy().item(),
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("total_correct >>>>>>>>>>>>>> ", total_correct)
        print("total_element >>>>>>>>>>>>>> ", total_element)
        print(
            "EP%d_%s, avg_loss=" % (epoch, str_code),
            avg_loss / len(data_iter),
            "total_acc=",
            total_correct * 100.0 / total_element,
        )

    def save(self, epoch, file_path="checkpoints"):
        """
        Saving the current Transformer model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "epoch%d" % epoch
        flow.save(self.trans.state_dict(), output_path)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

