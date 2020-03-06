import torch
import torch.nn.functional as F
import copy


class Rollout(object):
    def __init__(self, model, update_rate):
        self.ori_model = model
        self.rollout_model = copy.deepcopy(model)   # 使用该模型来进行roll-out
        self.update_rate = update_rate

    def get_reward(self, sample, num, discriminator):
        """

        :param sample: The output data
        :param num:
        :param discriminator:
        :return:
        """
        batch_size = sample.size(0)
        seq_len = sample.size(1)
        reward = []
        for i in range(num):
            for j in range(1, seq_len+1):
                temp_data = self.rollout_model.partial_sample(seq_len, data[:, :j])

                pred_reward = discriminator(temp_data)
                pred_reward = F.softmax(pred_reward, 1)

                if i == 0:
                    reward.append(pred_reward[:, 1].unsqueeze(1))
                else:
                    reward[j-1] += pred_reward[:, 1].unsqueeze(1)
        reward = torch.cat(reward, dim=1) / num
        return reward

    def update_param(self):
        """
        update the parameter with the the update_rate percent origin model.
        """
        for (name1, param1), (name2, param2) in \
                zip(self.ori_model.named_parameters(), self.rollout_model.named_parameters()):
            if name1 != name2:
                raise ValueError("The models parameter has been change")
            param1.data = self.update_rate * param1.data + (1 - self.update_rate) * param2.data

