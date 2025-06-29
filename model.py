import world
import torch
import torch.nn as nn
import utils

class PureBPR(nn.Module):
    def __init__(self, config, dataset):
        super(PureBPR, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        print("using Normal distribution initializer")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss


class LightGCN(nn.Module):
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self._init_weight()

    def _init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['layer']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()
        self.interactionGraph = self.dataset.getInteractionGraph()
        print(f"{world.model_name} is already to go")

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        G = self.interactionGraph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(G, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        self.final_user, self.final_item = users, items
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.final_user, self.final_item
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss


class DCSGCN(LightGCN):
    def _init_weight(self):
        super(DCSGCN, self)._init_weight()
        self.socialGraph = self.dataset.getSocialGraph()
        self.Graph_Comb = Graph_Comb(self.latent_dim)


        ##############################1,2.子图参数+节点分类##############################################    一样

        self.groups = 3
        self.fc = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.leaky = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(p=0.4)
        self.fc_g = torch.nn.Linear(self.latent_dim, self.groups)



    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        A = self.interactionGraph
        S = self.socialGraph



        ####################################3.子图生成##########################################   不一样


        temp = self.dropout(self.leaky(self.fc(all_emb)))    #E0    #########################################3
        group_scores = self.dropout(self.fc_g(temp))
        #group_scores = self.fc_g(temp)

        # 找出每个用户/物品所属的 group，生成 one-hot 编码
        a_top, a_top_idx = torch.topk(group_scores, k=1, sorted=False)   #k=1：第一大，不排序
        one_hot_emb = torch.eq(group_scores,a_top).float()               #最大分数处标1，其他位置标0
        # print('emb:',one_hot_emb.shape)

        u_one_hot, i_one_hot = torch.split(one_hot_emb, [self.num_users, self.num_items])
        # print(i_one_hot.shape)
        i_one_hot = torch.ones(i_one_hot.shape).to(world.device)         #物品分类向量全设为1
        one_hot_emb = torch.cat([u_one_hot, i_one_hot]).t()             #连接，并转置

        # print('oneemd:', one_hot_emb.shape,self.groups)

        # Create Subgraphs
        # 创建子图列表
        subgraph_list = []
        for g in range(self.groups):
            # print(A)
            temp = utils.cust_mul(A, one_hot_emb[g], 1)
            temp = utils.cust_mul(temp, one_hot_emb[g], 0)
            subgraph_list.append(temp)




        embs = [all_emb]
        for layer in range(self.n_layers):
            # embedding from last layer
            users_emb, items_emb = torch.split(all_emb, [self.num_users, self.num_items])
            # social network propagation(user embedding)
            users_emb_social = torch.sparse.mm(S, users_emb)
            # user-item bi-network propagation(user and item embedding)




            if layer==0:

                all_emb_interaction = torch.sparse.mm(A, all_emb)


            else:
                ##########################################4.子图聚合################################################    不一样

                A_users_emb = [None for _ in range(self.groups)]       #存子嵌入的列表

                for g in range(self.groups):
                    A_users_emb[g] = torch.sparse.mm(subgraph_list[g], all_emb)


                #########################################5.嵌入组合###############################################   不一样

                all_emb_interaction = torch.mean(torch.stack(A_users_emb),0)









            #all_emb_interaction = torch.sparse.mm(A, all_emb)
            # get users_emb_interaction
            users_emb_interaction, items_emb_next = torch.split(all_emb_interaction, [self.num_users, self.num_items])
            # graph fusion model
            users_emb_next = self.Graph_Comb(users_emb_social, users_emb_interaction)
            all_emb = torch.cat([users_emb_next, items_emb_next])
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        final_embs = torch.mean(embs, dim=1)
        users, items = torch.split(final_embs, [self.num_users, self.num_items])
        self.final_user, self.final_item = users, items
        return users, items


class Graph_Comb(nn.Module):
    def __init__(self, embed_dim):
        super(Graph_Comb, self).__init__()
        self.att_x = nn.Linear(embed_dim, embed_dim, bias=False)
        self.att_y = nn.Linear(embed_dim, embed_dim, bias=False)
        self.comb = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x, y):
        h1 = torch.tanh(self.att_x(x))
        h2 = torch.tanh(self.att_y(y))
        output = self.comb(torch.cat((h1, h2), dim=1))
        output = output / output.norm(2)
        return output
