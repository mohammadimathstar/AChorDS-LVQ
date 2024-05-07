from torch.autograd import Function
from util.grassmann import compute_distances_on_grassmann_mdf

import torch


def rotate_data(xs, rotation_matrix, winner_ids, return_rotation_matrix=False):
    
    assert xs.ndim == 3, f"data should be of shape (batch_size, dim_of_data, dim_of_subspace), but it is of shape {xs.shape}"
    assert winner_ids.shape[1] == 2, f"There should only be two winners W^+- prototypes for each data. But now there are {winner_ids.shape[1]} winners."
    
    nbatch = xs.shape[0]
    Qwinners  = rotation_matrix[torch.arange(nbatch).unsqueeze(-1), winner_ids] # shape: (batch_size, 2, d, d)
    
    Qwinners1, Qwinners2 = Qwinners[:, 0], Qwinners[:, 1] # shape: (batch_size, d, d)
    rotated_xs1, rotated_xs2 = torch.bmm(xs, Qwinners1), torch.bmm(xs, Qwinners2) # shape: (batch_size, D, d)

    if return_rotation_matrix:
        return rotated_xs1, rotated_xs2, Qwinners1, Qwinners2
    return rotated_xs1, rotated_xs2

class ChordalPrototypeLayer(Function):

    @staticmethod
    def forward(ctx, xs_subspace, xprotos, relevances):

        # Compute distances between data and prototypes
        output = compute_distances_on_grassmann_mdf(
            xs_subspace,
            xprotos,
            'chordal',
            relevances,                        
        )
        
        ctx.save_for_backward(
            xs_subspace, xprotos, relevances,
            output['distance'], output['Q'], output['Qw'], output['canonicalcorrelation'])
        return output['distance'], output['Qw']

    @staticmethod
    def backward(ctx, grad_output, grad_qw):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        
        nbatch = grad_output.shape[0]             
        
        xs_subspace, xprotos, relevances, distances, Q, Qw, cc = ctx.saved_tensors
        diag_rel = torch.tile(
            relevances[0],
            (xprotos.shape[-2], 1)
        )
        
        # there are some example that their gradient (of loss) is zero
        # here we set their indices to negative (we do not use them for training)
        winner_ids = torch.stack([
            torch.nonzero(gd).T[0] if len(torch.nonzero(gd).T[0]) == 2 else torch.tensor([-1, -2]) for gd in torch.unbind(grad_output)
        ], dim=0)
        
        if len(torch.argwhere(winner_ids<0)) > 0:
            s = torch.argwhere((winner_ids > 0)[:, 0]).T[0]
            xs_subspace = xs_subspace[s]
            distances = distances[s]
            Q = Q[s] 
            Qw = Qw[s]
            cc = cc[s]
            winner_ids = winner_ids[s]
            nbatch = s.shape[0]
            
        
        # **********************************************
        # ********** gradient of prototypes ************
        # **********************************************
        
        # Rotate data points (based on winner prototypes)       
        rotated_xs1, rotated_xs2, Qwinners1, Qwinners2 = rotate_data(xs_subspace,
                                                                     Q,
                                                                     winner_ids,
                                                                     return_rotation_matrix=True)
        dist_grad1 = grad_output[torch.arange(nbatch), winner_ids[torch.arange(nbatch), 0]]
        dist_grad2 = grad_output[torch.arange(nbatch), winner_ids[torch.arange(nbatch), 1]]
        
        # gradient of prototypes
        grad_protos1 = - rotated_xs1 * diag_rel.unsqueeze(0) * dist_grad1.unsqueeze(-1).unsqueeze(-1) 
        grad_protos2 = - rotated_xs2 * diag_rel.unsqueeze(0) * dist_grad2.unsqueeze(-1).unsqueeze(-1)

        # **********************************************
        # ********** gradient of relevances ************
        # **********************************************        
        CanCorrwinners1 = cc[torch.arange(nbatch), winner_ids[torch.arange(nbatch), 0]]
        CanCorrwinners2 = cc[torch.arange(nbatch), winner_ids[torch.arange(nbatch), 1]]
        grad_rel = - (
            CanCorrwinners1 * dist_grad1.unsqueeze(-1) + 
            CanCorrwinners2 * dist_grad2.unsqueeze(-1)
        )
        
        grad_xs = grad_protos = grad_relevances = None
        
        if ctx.needs_input_grad[0]: # TODO: set input grad to false
            print("Hey why the input need gradient, please check it!")
        if ctx.needs_input_grad[1]:
            grad_protos = torch.zeros_like(xprotos)
            grad_protos[winner_ids[torch.arange(nbatch), 0]] = grad_protos1.to(grad_protos.dtype)
            grad_protos[winner_ids[torch.arange(nbatch), 1]] = grad_protos2.to(grad_protos.dtype)
        if ctx.needs_input_grad[2]:
            grad_relevances = grad_rel

        return grad_xs, grad_protos, grad_relevances


