import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VALID_ACTIVATION = [a for a in dir(nn.modules.activation)
                    if not a.startswith('__')
                    and a not in ['torch', 'warnings', 'F', 'Parameter', 'Module']]
VALID_BATCHNORM_DIM = {1, 2, 3}


class DepthwiseSeperableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DepthwiseSeperableConv1d, self).__init__()

        self.depthwise_conv1d = nn.Conv1d(
            in_channels, in_channels, kernel_size, groups=in_channels,
            padding=kernel_size // 2)
        self.pointwise_conv1d = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv1d(x)
        x = self.pointwise_conv1d(x)

        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_size, output_size, n_heads, max_pos_distance, dropout,
                 pos_embedding_K, pos_embedding_V):
        super(MultiHeadSelfAttention, self).__init__()

        if output_size % n_heads != 0:
            raise ValueError(
                f'MultiHeadSelfAttention: output_size({output_size}) isn\'t'
                f'a multiplier of n_heads({n_heads})')

        self.output_size = output_size
        self.n_heads = n_heads
        self.d_head = output_size // n_heads
        self.max_pos_distance = max_pos_distance

        self.pos_embedding_K = pos_embedding_K
        self.pos_embedding_V = pos_embedding_V

        self.input_linear = nn.Linear(input_size, output_size * 3)
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.output_linear = nn.Linear(output_size, output_size)

    def forward(self, x, mask):
        batch_size, input_len, *_ = x.shape

        Q, K, V = self.input_linear(x).chunk(3, dim=-1)
        Q, K, V = [
            x.reshape(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
            for x in (Q, K, V)
        ]

        device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        pos_index = torch.arange(input_len).reshape(1, -1).repeat(input_len, 1)
        pos_index = pos_index - pos_index.t()
        pos_index = pos_index.clamp(-self.max_pos_distance, self.max_pos_distance)
        pos_index += self.max_pos_distance
        pos_index = pos_index.to(dtype=torch.int64, device=device)

        # TODO?: add dropout to position embedding
        # calculate attention score (relative position representation #1)
        S1 = Q @ K.transpose(-1, -2)
        Q = Q.reshape(-1, input_len, self.d_head).transpose(0, 1)
        pos_emb_K = self.pos_embedding_K(pos_index)
        S2 = (Q @ pos_emb_K.transpose(-1, -2)).transpose(0, 1)
        S2 = S2.reshape(batch_size, self.n_heads, input_len, input_len)
        S = (S1 + S2) / np.sqrt(self.d_head)

        # set score of V padding tokens to 0
        S.masked_fill_(mask.reshape(batch_size, 1, 1, -1) == 0, -np.inf)
        A = F.softmax(S, dim=-1)
        if self.dropout:
            A = self.dropout(A)

        # apply attention to get output (relative position representation #2)
        O1 = A @ V
        A = A.reshape(-1, input_len, input_len).transpose(0, 1)
        pos_emb_V = self.pos_embedding_V(pos_index)
        O2 = (A @ pos_emb_V).transpose(0, 1)
        O2 = O2.reshape(batch_size, self.n_heads, input_len, self.d_head)
        output = O1 + O2
        output = output.transpose(1, 2).reshape(batch_size, -1, self.output_size)
        output = self.output_linear(output)

        return output


class PointwiseFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(PointwiseFeedForward, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.activation = Activation('ReLU')
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.linear2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.linear2(x)

        return x


class Activation(nn.Module):
    def __init__(self, activation, *args, **kwargs):
        super(Activation, self).__init__()

        if activation in VALID_ACTIVATION:
            self.activation = \
                getattr(nn.modules.activation, activation)(*args, **kwargs)
        else:
            raise ValueError(
                f'Activation: {activation} is not a valid activation function')

    def forward(self, x):
        return self.activation(x)


class BatchNormResidual(nn.Module):
    def __init__(self, sublayer, n_features, dim=1, transpose=False, activation=None,
                 dropout=0):
        super(BatchNormResidual, self).__init__()

        self.sublayer = sublayer
        if dim in VALID_BATCHNORM_DIM:
            batch_norm = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
            self.batch_norm = batch_norm[dim - 1](n_features)
        else:
            raise ValueError(
                f'BatchNormResidual: dim must be one of {{1, 2, 3}}, but got {dim}')
        self.transpose = transpose
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout else None

    def forward(self, x, *args, **kwargs):
        """
        The operations are ordered according to https://arxiv.org/abs/1804.09849.
        """
        # Normalize
        if self.transpose:
            y = x.transpose(1, 2).contiguous()
        else:
            y = x
        y = self.batch_norm(y)
        if self.transpose:
            y = y.transpose(1, 2)

        # Transform
        y = self.sublayer(x, *args, **kwargs)
        if self.activation:
            y = self.activation(y)

        # Dropout
        if self.dropout:
            y = self.dropout(y)

        # Residual
        y += x

        return y


class LayerNormResidual(nn.Module):
    def __init__(self, sublayer, norm_shape, transpose=False, activation=None,
                 dropout=0):
        super(LayerNormResidual, self).__init__()

        self.sublayer = sublayer
        self.layer_norm = nn.LayerNorm(norm_shape)
        self.transpose = transpose
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout else None

    def forward(self, x, *args, **kwargs):
        """
        The operations are ordered according to https://arxiv.org/abs/1804.09849.
        """
        # Normalize
        if self.transpose:
            y = x.transpose(1, 2).contiguous()
        else:
            y = x
        y = self.layer_norm(y)
        if self.transpose:
            y = y.transpose(1, 2)

        # Transform
        y = self.sublayer(x, *args, **kwargs)
        if self.activation:
            y = self.activation(y)

        # Dropout
        if self.dropout:
            y = self.dropout(y)

        # Residual
        y += x

        return y


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_convs, kernel_size, n_heads,
                 max_pos_distance, dropout, pos_embedding_K, pos_embedding_V):
        super(EncoderBlock, self).__init__()

        # self.convs = nn.ModuleList([
        #     BatchNormResidual(
        #         DepthwiseSeperableConv1d(d_model, d_model, kernel_size), d_model,
        #         activation=Activation('ReLU'), dropout=dropout)
        #     for _ in range(n_convs)
        # ])
        # self.attention = BatchNormResidual(
        #     MultiHeadSelfAttention(
        #         d_model, d_model, n_heads, max_pos_distance, dropout, pos_embedding_K,
        #         pos_embedding_V),
        #     d_model, transpose=True, dropout=dropout)
        # self.feedforward = BatchNormResidual(
        #     PointwiseFeedForward(d_model, d_model * 4, dropout), d_model,
        #     transpose=True, activation=Activation('ReLU'), dropout=dropout)
        self.convs = nn.ModuleList([
            LayerNormResidual(
                DepthwiseSeperableConv1d(d_model, d_model, kernel_size), d_model,
                transpose=True, activation=Activation('ReLU'), dropout=dropout)
            for _ in range(n_convs)
        ])
        self.attention = LayerNormResidual(
            MultiHeadSelfAttention(
                d_model, d_model, n_heads, max_pos_distance, dropout, pos_embedding_K,
                pos_embedding_V),
            d_model, dropout=dropout)
        self.feedforward = LayerNormResidual(
            PointwiseFeedForward(d_model, d_model * 4, dropout), d_model,
            activation=Activation('ReLU'), dropout=dropout)

    def forward(self, x, x_pad_mask):
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2)

        x = self.attention(x, x_pad_mask)
        x = self.feedforward(x)

        return x


class Encoder(nn.Module):
    def __init__(self, n_blocks, input_size, d_model, n_convs, kernel_size,
                 n_heads, max_pos_distance, pos_embedding_K, pos_embedding_V, dropout):
        super(Encoder, self).__init__()

        self.conv = DepthwiseSeperableConv1d(input_size, d_model, kernel_size)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = Activation('ReLU')
        self.dropout = nn.Dropout(p=dropout) if dropout else None

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                d_model, n_convs, kernel_size, n_heads, max_pos_distance, dropout,
                pos_embedding_K, pos_embedding_V
            )
            for i in range(n_blocks)
        ])

    def forward(self, x, x_pad_mask):
        x = x.transpose(1, 2)
        x = self.batch_norm(self.conv(x))
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        x = x.transpose(1, 2)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, x_pad_mask)

        return x


class Coattention(nn.Module):
    def __init__(self, input_size):
        super(Coattention, self).__init__()

        self.linear = nn.Linear(input_size * 3, 1)

    def forward(self, C, Q, C_mask, Q_mask):
        c_len, q_len = C.shape[1], Q.shape[1]

        C_expanded = C.unsqueeze(2).expand(-1, -1, q_len, -1)
        Q_expanded = Q.unsqueeze(1).expand(-1, c_len, -1, -1)

        x = torch.cat((C_expanded, Q_expanded, C_expanded * Q_expanded), dim=-1)
        S = self.linear(x).squeeze(-1)
        S_row = S.masked_fill(Q_mask.unsqueeze(1) == 0, -np.inf)
        S_row = F.softmax(S_row, dim=2)
        S_col = S.masked_fill(C_mask.unsqueeze(2) == 0, -np.inf)
        S_col = F.softmax(S_col, dim=1)

        A = S_row @ Q
        B = S_row @ S_col.transpose(1, 2) @ C
        CQ = torch.cat((C, A, C * A, C * B), dim=-1)

        return CQ


class QANet(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_blocks, d_model, n_convs, kernel_size,
                 n_heads, max_pos_distance, dropout):
        super(QANet, self).__init__()

        pos_embedding_K = nn.Embedding(2 * max_pos_distance + 1, d_model // n_heads)
        pos_embedding_V = nn.Embedding(2 * max_pos_distance + 1, d_model // n_heads)
        self.C_encoder = Encoder(
            n_blocks, emb_dim, d_model, n_convs, kernel_size, n_heads,
            max_pos_distance, pos_embedding_K, pos_embedding_V, dropout)
        self.Q_encoder = Encoder(
            n_blocks, emb_dim, d_model, n_convs, kernel_size, n_heads,
            max_pos_distance, pos_embedding_K, pos_embedding_V, dropout)
        self.coattention = Coattention(d_model)
        self.M_linear = nn.Linear(d_model * 4, d_model)
        self.M_encoder = Encoder(
            n_blocks, d_model, d_model, n_convs, kernel_size, n_heads,
            max_pos_distance, pos_embedding_K, pos_embedding_V, dropout)
        # self.start_linear = nn.Linear(d_model * 2, 1)
        # self.end_linear = nn.Linear(d_model * 2, 1)
        self.start_linear = nn.Linear(d_model, 1)
        self.end_linear = nn.Linear(d_model, 1)

    def forward(self, C, Q, C_pad_mask, Q_pad_mask):
        C = self.C_encoder(C, C_pad_mask)
        Q = self.Q_encoder(Q, Q_pad_mask)
        CQ = self.M_linear(self.coattention(C, Q, C_pad_mask, Q_pad_mask))
        M0 = self.M_encoder(CQ, C_pad_mask)
        # M1 = self.M_encoder(M0, C_pad_mask)
        # M2 = self.M_encoder(M1, C_pad_mask)
        # start_logits = self.start_linear(torch.cat((M0, M1), dim=-1)).squeeze(-1)
        # end_logits = self.end_linear(torch.cat((M0, M2), dim=-1)).squeeze(-1)
        start_logits = self.start_linear(M0).squeeze(-1)
        end_logits = self.end_linear(M0).squeeze(-1)

        return start_logits, end_logits
