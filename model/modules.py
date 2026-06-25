import pyro
import torch
import torch.nn as nn
import torch.nn.functional as F


def coord2radial(edge_index, coord):
    row, col = edge_index
    coord_diff = coord[row] - coord[col]  # [n_edge, n_channel, d]
    radial = torch.bmm(coord_diff, coord_diff.transpose(-1, -2))  # [n_edge, n_channel, n_channel]
    # normalize radial
    radial = F.normalize(radial, dim=0)  # [n_edge, n_channel, n_channel]
    return radial, coord_diff


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res


def unsorted_segment_sum(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class RelationMPNN(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, n_channel, dropout=0.1, edges_in_d=1, edge_type=8):
        super(RelationMPNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.coord_mlp = nn.ModuleList()
        self.relation_mlp = nn.ModuleList()

        self.message_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + n_channel**2 + edges_in_d, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU())

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_nf + edges_in_d + hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, edges_in_d))
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, output_nf))
         
        for _ in range(edge_type):
            self.relation_mlp.append(nn.Linear(input_nf, input_nf, bias=False))

            layer = nn.Linear(hidden_nf, n_channel, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

            self.coord_mlp.append(nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf),
                nn.SiLU(),
                layer
            ))

    def message_model(self, source, target, radial, edge_attr):
        # Compute explicit dimension to handle empty edge case (0 edges)
        n_edges = radial.shape[0]
        radial_dim = radial.shape[1] * radial.shape[2] if len(radial.shape) > 2 else radial.shape[1]
        radial = radial.reshape(n_edges, radial_dim)
        out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.message_mlp(out)
        out = self.dropout(out)

        return out

    def node_model(self, x, edge_list, edge_feat_list, sampled_index_list):
        agg = self.relation_mlp[0](unsorted_segment_sum(edge_feat_list[0], edge_list[0][0], num_segments=x.size(0)))
        for i in range(1, len(edge_list)):
            if i == 6 and (sampled_index_list is not None):
                agg += self.relation_mlp[i](unsorted_segment_sum(edge_feat_list[i][sampled_index_list[0]], edge_list[i][0][sampled_index_list[0]], num_segments=x.size(0)))
            elif i == 7 and (sampled_index_list is not None):
                agg += self.relation_mlp[i](unsorted_segment_sum(edge_feat_list[i][sampled_index_list[1]], edge_list[i][0][sampled_index_list[1]], num_segments=x.size(0)))
            else:
                agg += self.relation_mlp[i](unsorted_segment_sum(edge_feat_list[i], edge_list[i][0], num_segments=x.size(0)))
            
        agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        out = self.dropout(out)
        out = x + out

        return out
    
    def coord_model(self, coord, edge_list, edge_feat_list, coord_diff_list, segment_ids):
        tran_list = []
        row_list = []
        if segment_ids is None:
            sampled_index_list = None
        else:
            sampled_index_list = []

        for i in range(len(edge_list)):
            trans = coord_diff_list[i] * self.coord_mlp[i](edge_feat_list[i]).unsqueeze(-1)  # [n_edge, n_channel, d]
            edges = edge_list[i][0]
            if (i == 6 or i == 7) and (segment_ids is not None):
                antigen_edge_list = sequential_or(segment_ids[edge_list[i][0]] == 3, segment_ids[edge_list[i][1]] == 3)
                sampled_index = torch.ones(trans.shape[0]).to(trans.device)
                if antigen_edge_list.sum() != 0:
                    weight = torch.abs(self.coord_mlp[i](edge_feat_list[i]).mean(dim=-1))[antigen_edge_list]
                    sampled_index[antigen_edge_list] = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=0.5, probs=(weight  - weight.min()) / (weight.max() - weight.min())).rsample()
                sampled_index = sampled_index.bool()
                trans = trans[sampled_index]
                edges = edge_list[i][0][sampled_index]
                sampled_index_list.append(sampled_index)
            tran_list.append(trans)
            row_list.append(edges)
        agg = unsorted_segment_mean(torch.cat(tran_list, dim=0), torch.cat(row_list, dim=0), num_segments=coord.size(0))  # [bs * n_node, n_channel, d]
        coord = coord + agg

        return coord, sampled_index_list

    def edge_model(self, h, edge_list, edge_feat_list):
        m = []

        for i in range(len(edge_list)):
            row, col = edge_list[i]
            out = torch.cat([h[row], edge_feat_list[i], h[col]], dim=1)
            out = self.edge_mlp(out)
            m.append(out)

        return m
    
    def forward(self, h, coord, edge_attr, edge_list, segment_ids=None):

        edge_feat_list = []
        coord_diff_list = []

        for i in range(len(edge_list)):
            radial, coord_diff = coord2radial(edge_list[i], coord)
            coord_diff_list.append(coord_diff)

            row, col = edge_list[i]
            edge_feat = self.message_model(h[row], h[col], radial, edge_attr[i])
            edge_feat_list.append(edge_feat)

        x, sampled_index_list = self.coord_model(coord, edge_list, edge_feat_list, coord_diff_list, segment_ids)
        h = self.node_model(h, edge_list, edge_feat_list, sampled_index_list)
        m = self.edge_model(h, edge_list, edge_attr)

        return h, x, m


class VirtualNodeMPNN(nn.Module):
    """MPNN layer with virtual node support.

    Virtual nodes connect to ALL epitope and CDR nodes via dedicated edge types.
    This bypasses over-squashing by aggregating interface info into virtual nodes
    and broadcasting back to real nodes.

    Edge types (total 10):
        0-7: Standard 8 relation types from RelationMPNN
        8: vn_to_epitope (vn <-> epitope bidirectional)
        9: vn_to_cdr (vn <-> CDR bidirectional)
    """

    def __init__(self, input_nf, output_nf, hidden_nf, n_channel, dropout=0.1,
                 edges_in_d=1, edge_type=10):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.n_channel = n_channel
        self.hidden_nf = hidden_nf
        self.edge_type = edge_type

        # Standard message MLP (for real-to-real edges with radial features)
        self.message_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + n_channel**2 + edges_in_d, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU())

        # Virtual node message MLP (no radial -- virtual nodes have no physical coords)
        self.vn_message_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + edges_in_d, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU())

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_nf + edges_in_d + hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, edges_in_d))

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, output_nf))

        # Relation MLPs for all edge types (8 standard + 2 virtual node types)
        self.relation_mlp = nn.ModuleList()
        self.coord_mlp = nn.ModuleList()
        for i in range(edge_type):
            self.relation_mlp.append(nn.Linear(input_nf, input_nf, bias=False))

            layer = nn.Linear(hidden_nf, n_channel, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
            self.coord_mlp.append(nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf),
                nn.SiLU(),
                layer
            ))

    def message_model(self, source, target, radial, edge_attr):
        radial = radial.reshape(radial.shape[0], -1)
        out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.message_mlp(out)
        out = self.dropout(out)
        return out

    def vn_message_model(self, source, target, edge_attr):
        """Message model for virtual node edges (no radial info)."""
        out = torch.cat([source, target, edge_attr], dim=1)
        out = self.vn_message_mlp(out)
        out = self.dropout(out)
        return out

    def node_model(self, x, edge_list, edge_feat_list, sampled_index_list, n_real_nodes):
        """Node update. Aggregates from all edge types including virtual node edges."""
        if edge_list[0].numel() > 0:
            agg = self.relation_mlp[0](unsorted_segment_sum(
                edge_feat_list[0], edge_list[0][0], num_segments=x.size(0)))
        else:
            agg = torch.zeros(x.size(0), x.size(1), device=x.device)

        for i in range(1, len(edge_list)):
            if edge_list[i].numel() == 0:
                continue
            if i == 6 and sampled_index_list is not None and len(sampled_index_list) > 0:
                agg = agg + self.relation_mlp[i](unsorted_segment_sum(
                    edge_feat_list[i][sampled_index_list[0]],
                    edge_list[i][0][sampled_index_list[0]],
                    num_segments=x.size(0)))
            elif i == 7 and sampled_index_list is not None and len(sampled_index_list) > 1:
                agg = agg + self.relation_mlp[i](unsorted_segment_sum(
                    edge_feat_list[i][sampled_index_list[1]],
                    edge_list[i][0][sampled_index_list[1]],
                    num_segments=x.size(0)))
            else:
                agg = agg + self.relation_mlp[i](unsorted_segment_sum(
                    edge_feat_list[i], edge_list[i][0], num_segments=x.size(0)))

        agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        out = self.dropout(out)
        out = x + out
        return out

    def coord_model(self, coord, edge_list, edge_feat_list, coord_diff_list,
                    segment_ids, n_real_nodes):
        """Coordinate update. Only updates real nodes (not virtual nodes)."""
        tran_list = []
        row_list = []
        if segment_ids is None:
            sampled_index_list = None
        else:
            sampled_index_list = []

        for i in range(min(8, len(edge_list))):
            if edge_list[i].numel() == 0:
                continue
            trans = coord_diff_list[i] * self.coord_mlp[i](edge_feat_list[i]).unsqueeze(-1)
            edges = edge_list[i][0]

            if (i == 6 or i == 7) and segment_ids is not None:
                seg_ids_for_edges = segment_ids[:n_real_nodes] if n_real_nodes < len(segment_ids) else segment_ids
                real_edge_mask = (edge_list[i][0] < n_real_nodes) & (edge_list[i][1] < n_real_nodes)
                if real_edge_mask.any():
                    real_row = edge_list[i][0][real_edge_mask]
                    real_col = edge_list[i][1][real_edge_mask]
                    antigen_edge_list = sequential_or(
                        seg_ids_for_edges[real_row] == 3,
                        seg_ids_for_edges[real_col] == 3)
                    sampled_index = torch.ones(trans.shape[0], device=trans.device)
                    if antigen_edge_list.sum() != 0:
                        weight = torch.abs(self.coord_mlp[i](edge_feat_list[i]).mean(dim=-1))[real_edge_mask][antigen_edge_list]
                        denom = weight.max() - weight.min()
                        if denom < 1e-8 or torch.isnan(denom):
                            probs = torch.full_like(weight, 0.5)
                        else:
                            probs = ((weight - weight.min()) / denom).clamp(1e-6, 1 - 1e-6)
                        sampled_index_sub = torch.ones(real_edge_mask.sum(), device=trans.device)
                        sampled_index_sub[antigen_edge_list] = pyro.distributions.RelaxedBernoulliStraightThrough(
                            temperature=0.5, probs=probs).rsample()
                        sampled_index[real_edge_mask] = sampled_index_sub
                    sampled_index = sampled_index.bool()
                    trans = trans[sampled_index]
                    edges = edge_list[i][0][sampled_index]
                    sampled_index_list.append(sampled_index)
                else:
                    tran_list.append(trans)
                    row_list.append(edges)
                    continue
            tran_list.append(trans)
            row_list.append(edges)

        if tran_list:
            all_trans = torch.cat(tran_list, dim=0)
            all_rows = torch.cat(row_list, dim=0)
            agg = unsorted_segment_mean(all_trans, all_rows, num_segments=coord.size(0))
            coord[:n_real_nodes] = coord[:n_real_nodes] + agg[:n_real_nodes]

        return coord, sampled_index_list

    def edge_model(self, h, edge_list, edge_feat_list):
        m = []
        for i in range(len(edge_list)):
            if edge_list[i].numel() == 0:
                m.append(edge_feat_list[i])
                continue
            row, col = edge_list[i]
            out = torch.cat([h[row], edge_feat_list[i], h[col]], dim=1)
            out = self.edge_mlp(out)
            m.append(out)
        return m

    def forward(self, h, coord, edge_attr, edge_list, segment_ids=None,
                n_real_nodes=None, vn_edge_types=(8, 9)):
        if n_real_nodes is None:
            n_real_nodes = h.size(0)

        edge_feat_list = []
        coord_diff_list = []

        for i in range(len(edge_list)):
            if edge_list[i].numel() == 0:
                edge_feat_list.append(torch.zeros(0, self.hidden_nf, device=h.device))
                coord_diff_list.append(torch.zeros(0, self.n_channel, 3, device=h.device))
                continue

            if i in vn_edge_types:
                row, col = edge_list[i]
                edge_feat = self.vn_message_model(h[row], h[col], edge_attr[i])
                edge_feat_list.append(edge_feat)
                coord_diff_list.append(torch.zeros(
                    edge_list[i].shape[1], self.n_channel, 3, device=h.device))
            else:
                radial, coord_diff = coord2radial(edge_list[i], coord)
                coord_diff_list.append(coord_diff)
                row, col = edge_list[i]
                edge_feat = self.message_model(h[row], h[col], radial, edge_attr[i])
                edge_feat_list.append(edge_feat)

        x, sampled_index_list = self.coord_model(
            coord, edge_list, edge_feat_list, coord_diff_list, segment_ids, n_real_nodes)
        h = self.node_model(h, edge_list, edge_feat_list, sampled_index_list, n_real_nodes)
        m = self.edge_model(h, edge_list, edge_attr)

        return h, x, m


class VirtualNodeEGNN(nn.Module):
    """EGNN with virtual interface nodes for bypassing over-squashing.

    Virtual nodes (N_vn=3 by default) connect to ALL epitope and CDR nodes.
    They aggregate interface information and broadcast it back, creating
    shortcut paths that bypass the standard message-passing bottleneck.
    """

    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel, n_layers=4,
                 dropout=0.1, node_feats_dim=0, edge_feats_dim=1, n_virtual_nodes=3):
        super().__init__()
        self.n_layers = n_layers
        self.n_virtual_nodes = n_virtual_nodes
        self.hidden_nf = hidden_nf
        self.n_channel = n_channel
        self.dropout = nn.Dropout(dropout)

        self.linear_in = nn.Linear(in_node_nf + node_feats_dim, hidden_nf)
        self.linear_out = nn.Linear(hidden_nf, out_node_nf)

        self.vn_init_h = nn.Parameter(torch.randn(n_virtual_nodes, hidden_nf) * 0.02)
        self.vn_init_pos = nn.Parameter(torch.zeros(n_virtual_nodes, n_channel, 3))

        self.vn_edge_emb = nn.ParameterDict({
            'vn_to_epi': nn.Parameter(torch.randn(1, edge_feats_dim) * 0.02),
            'vn_to_cdr': nn.Parameter(torch.randn(1, edge_feats_dim) * 0.02),
        })

        for i in range(n_layers):
            self.add_module(f'layer_{i}', VirtualNodeMPNN(
                hidden_nf, hidden_nf, hidden_nf, n_channel,
                dropout=dropout, edges_in_d=edge_feats_dim, edge_type=10))

    def _build_vn_edges(self, n_real_nodes, epitope_mask, cdr_mask, device):
        n_vn = self.n_virtual_nodes
        vn_indices = torch.arange(n_real_nodes, n_real_nodes + n_vn, device=device)

        epi_indices = torch.where(epitope_mask)[0]
        if epi_indices.numel() > 0:
            vn_src = vn_indices.repeat_interleave(epi_indices.numel())
            epi_dst = epi_indices.repeat(n_vn)
            epi_src = epi_dst.clone()
            vn_dst = vn_src.clone()
            vn_to_epi_edges = torch.stack([
                torch.cat([vn_src, epi_src]),
                torch.cat([epi_dst, vn_dst])
            ])
        else:
            vn_to_epi_edges = torch.zeros(2, 0, dtype=torch.long, device=device)

        cdr_indices = torch.where(cdr_mask)[0]
        if cdr_indices.numel() > 0:
            vn_src = vn_indices.repeat_interleave(cdr_indices.numel())
            cdr_dst = cdr_indices.repeat(n_vn)
            cdr_src = cdr_dst.clone()
            vn_dst = vn_src.clone()
            vn_to_cdr_edges = torch.stack([
                torch.cat([vn_src, cdr_src]),
                torch.cat([cdr_dst, vn_dst])
            ])
        else:
            vn_to_cdr_edges = torch.zeros(2, 0, dtype=torch.long, device=device)

        return vn_to_epi_edges, vn_to_cdr_edges

    def forward(self, h, x, edges_list, edge_feats_list, node_feats, segment_ids,
                interface_only, epitope_mask, cdr_mask):
        device = h.device
        n_real_nodes = h.size(0)

        h = torch.cat((h, node_feats), 1)
        h = self.linear_in(h)
        h = self.dropout(h)

        h = torch.cat([h, self.vn_init_h.to(device)], dim=0)
        x = torch.cat([x, self.vn_init_pos.to(device)], dim=0)

        vn_to_epi_edges, vn_to_cdr_edges = self._build_vn_edges(
            n_real_nodes, epitope_mask, cdr_mask, device)

        extended_edges_list = list(edges_list) + [vn_to_epi_edges, vn_to_cdr_edges]

        n_epi_edges = vn_to_epi_edges.shape[1]
        n_cdr_edges = vn_to_cdr_edges.shape[1]
        vn_epi_feats = self.vn_edge_emb['vn_to_epi'].expand(n_epi_edges, -1)
        vn_cdr_feats = self.vn_edge_emb['vn_to_cdr'].expand(n_cdr_edges, -1)
        extended_edge_feats_list = list(edge_feats_list) + [vn_epi_feats, vn_cdr_feats]

        extended_seg_ids = torch.cat([
            segment_ids,
            torch.zeros(self.n_virtual_nodes, dtype=segment_ids.dtype, device=device)
        ])

        m = extended_edge_feats_list
        for i in range(self.n_layers):
            if interface_only == 0:
                h, x, m = self._modules[f'layer_{i}'](
                    h, x, m, extended_edges_list, segment_ids=None,
                    n_real_nodes=n_real_nodes, vn_edge_types=(8, 9))
            else:
                h, x, m = self._modules[f'layer_{i}'](
                    h, x, m, extended_edges_list, segment_ids=extended_seg_ids,
                    n_real_nodes=n_real_nodes, vn_edge_types=(8, 9))

        h_real = h[:n_real_nodes]
        x_real = x[:n_real_nodes]

        out = self.dropout(h_real)
        out = self.linear_out(out)

        return out, x_real, h_real


class RelationEGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel, n_layers=4, dropout=0.1, node_feats_dim=0, edge_feats_dim=1):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)

        self.linear_in = nn.Linear(in_node_nf + node_feats_dim, hidden_nf)
        self.linear_out = nn.Linear(hidden_nf, out_node_nf)

        for i in range(n_layers):
            self.add_module(f'layer_{i}', RelationMPNN(hidden_nf, hidden_nf, hidden_nf, n_channel, dropout=dropout, edges_in_d=edge_feats_dim))
    
    def forward(self, h, x, edges_list, edge_feats_list, node_feats, segment_ids, interface_only):
        h = torch.cat((h, node_feats), 1)
        h = self.linear_in(h)
        h = self.dropout(h)

        m = edge_feats_list
        for i in range(self.n_layers):
            if interface_only == 0:
                h, x, m = self._modules[f'layer_{i}'](h, x, m, edges_list)
            else:
                h, x, m = self._modules[f'layer_{i}'](h, x, m, edges_list, segment_ids)

        out = self.dropout(h)
        out = self.linear_out(out)

        return out, x, h