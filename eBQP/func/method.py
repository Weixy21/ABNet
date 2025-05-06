
import torch


def eBQP(H, F, A, b):

    # H is set to an identity matix in this function
    # H: batch x q x q,  q is the dimension of decision variable
    # F: batch x q
    # A: batch x n x q,  n is the number of constraint
    # b: batch x n

    G1 = A[:,0,:]
    G2 = A[:,1,:]
    h1 = b[:,0:1]
    h2 = b[:,1:2]

    
    y1_bar = 1*G1  # H or Q = identity matrix
    y2_bar = 1*G2
    u_bar = -F     # reference control
    p1_bar = h1 - torch.sum(G1*u_bar,dim = 1).unsqueeze(1)
    p2_bar = h2 - torch.sum(G2*u_bar,dim = 1).unsqueeze(1)

    G = torch.cat([torch.sum(y1_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y1_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0)], dim = 0)
    #G = 1*[y1_bar*y1_bar', y1_bar*y2_bar'; y2_bar*y1_bar', y2_bar*y2_bar']
    w_p1_bar = torch.clamp(p1_bar, max=0)
    w_p2_bar = torch.clamp(p2_bar, max=0)

    # G 0-(1,1), 1-(1,2), 2-(2,1), 3-(2,2)
    lambda1 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, torch.zeros_like(p1_bar), torch.where(G[1]*w_p1_bar < G[0]*p2_bar, w_p1_bar/G[0], torch.clamp(G[3]*p1_bar - G[2]*p2_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))
    
    lambda2 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, w_p2_bar/G[3], torch.where(G[1]*w_p1_bar < G[0]*p2_bar, torch.zeros_like(p1_bar), torch.clamp(G[0]*p2_bar - G[1]*p1_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))

    x = lambda1*y1_bar + lambda2*y2_bar + u_bar

    return x

