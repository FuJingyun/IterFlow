import torch


class Graph:
    def __init__(self, edges, edge_feats, k_neighbors, size):
        """
        Directed nearest neighbor graph constructed on a point cloud.

        Parameters
        ----------
        edges : torch.Tensor
            Contains list with nearest neighbor indices.
        edge_feats : torch.Tensor
            Contains edge features: relative point coordinates.
        k_neighbors : int
            Number of nearest neighbors.
        size : tuple(int, int)
            Number of points.

        """
        

        self.edges = edges
        self.size = tuple(size)
        self.edge_feats = edge_feats
        self.k_neighbors = k_neighbors

    @staticmethod
    def construct_graph(pcloud, nb_neighbors):
        """
        Construct a directed nearest neighbor graph on the input point cloud.

        Parameters
        ----------
        pcloud : torch.Tensor
            Input point cloud. Size B x N x 3.
        nb_neighbors : int
            Number of nearest neighbors per point.

        Returns
        -------
        graph : flot.models.graph.Graph
            Graph build on input point cloud containing the list of nearest 
            neighbors (NN) for each point and all edge features (relative 
            coordinates with NN).
            
        """

        # Size
        nb_points = pcloud.shape[1]
        size_batch = pcloud.shape[0]

        # Distance between points
        distance_matrix = torch.sum(pcloud ** 2, -1, keepdim=True)
        #　相加时，把所有为1的轴进行复制扩充，从而得到两个维度完全相同的张量。然后对应位置相加即可
        distance_matrix = distance_matrix + distance_matrix.transpose(1, 2)
        distance_matrix = distance_matrix - 2 * torch.bmm(
            pcloud, pcloud.transpose(1, 2)
        )

        # Find nearest neighbors
        # 升序，从小到大
        # distance_matrix：B x N x N
        # 只留下距离最近的 nb_neighbors 个邻近点
        # neighbors：B x N x nb_neighbors
        neighbors = torch.argsort(distance_matrix, -1)[..., :nb_neighbors]
        # effective_nb_neighbors = nb_neighbors 为一个数32
        effective_nb_neighbors = neighbors.shape[-1]
        # neighbors： B x ( N  nb_neighbors)
        neighbors = neighbors.reshape(size_batch, -1)

        # Edge origin
        idx = torch.arange(nb_points, device = distance_matrix.device).long()
        # idx [0 0 0 0 ....0 1 1 1 ... 1 ....]
        # 长度 nb_points x nb_neighbors
        idx = torch.repeat_interleave(idx, effective_nb_neighbors)

        # Edge features
        edge_feats = []
        for ind_batch in range(size_batch):
            edge_feats.append(
                #  维度 ( N  nb_neighbors 3)
                pcloud[ind_batch, neighbors[ind_batch]] - pcloud[ind_batch, idx]
            )

        # edge_feats： B N nb_neighbors x 3
        edge_feats = torch.cat(edge_feats, 0)

        # Handle batch dimension to get indices of nearest neighbors
        for ind_batch in range(1, size_batch):
            # 多个bacth合并后是第几个点
            neighbors[ind_batch] = neighbors[ind_batch] + ind_batch * nb_points
        # 平铺为一维 B   N  nb_neighbors
        neighbors = neighbors.view(-1)

        # Create graph

        '''
        neighbors: 每个点的邻居的index 序列,每个点的邻居在多个bacth合并后是第几个点,平铺为一维 B * N * nb_neighbors
        edge_feats:二维 B ,  N * nb_neighbors x 3 ,每个点的各个邻居到该点的相对坐标
        '''

        # neighbors平铺为一维 B   N  nb_neighbors
        # edge_feats： B x ( N  nb_neighbors)
        # effective_nb_neighbors = nb_neighbors 为一个数32
        graph = Graph(
            neighbors,
            edge_feats,
            effective_nb_neighbors,
            [size_batch * nb_points, size_batch * nb_points],
        )

        return graph

