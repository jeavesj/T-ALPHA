from torch import nn

from src.models.egnn import EGNN


class EGNN_Encoder(nn.Module):
    def __init__(
        self,
        in_node_nf=27,
        hidden_nf=64,
        out_node_nf=64,
        in_edge_nf=6,
        device="cuda",
        act_fn=nn.SiLU(),
        n_layers=12,
        residual=True,
        attention=True,
        normalize=False,
        tanh=False,
    ):
        super(EGNN_Encoder, self).__init__()

        # Initialize the EGNN model
        self.egnn = EGNN(
            in_node_nf,
            hidden_nf,
            out_node_nf,
            in_edge_nf,
            device,
            act_fn,
            n_layers,
            residual,
            attention,
            normalize,
            tanh,
        )

    def forward(self, h, x, edges, edge_attr):
        # Pass through EGNN layers to get updated node features and coordinates
        h, x = self.egnn(h, x, edges, edge_attr)

        return h
