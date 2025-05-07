
import torch


def eBQP(H, F, A, b):

    ''' solve a batch of QP in closed-form, special case: H is identity, and the number of constraint is 2
    The QP for each instance in the batch is of the form

                x* =   argmin_x 1/2 x^T H x + F^T x
                        subject to Ax <= b

            where H \in I^{nBatch, q, q}, q is the dimension of decision variable x
                I^{nBatch, q, q} is a batch of identity matrix,
                F \in R^{nBatch, q}
                A \in R^{nBatch, 2, q}
                b \in R^{nBatch, 2}

    Parameters:
            H:  A (nBatch, q, q)  Tensor.
            F:  A (nBatch, q) Tensor.
            A:  A (nBatch, 2, q) Tensor.
            b:  A (nBatch, 2) Tensor.

    Returns: x*: a (nBatch, q) Tensor.                       
    
    '''

    _, nConstraint = b.size()

    if nConstraint != 2:
        raise RuntimeError('The number of constraints should be two, use eBQP_g instead if the number of constraint is greater than 2')

    G1 = A[:,0,:]
    G2 = A[:,1,:]
    h1 = b[:,0:1]
    h2 = b[:,1:2]

    
    y1_bar = 1*G1  # H = identity matrix
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


def eBQP_I(H, F, A, b, barrier = [], option = 'min'):

    ''' solve a batch of QP in closed-form, special case: H is identity
    The QP for each instance in the batch is of the form

                x* =   argmin_x 1/2 x^T H x + F^T x
                        subject to Ax <= b

            where H \in S^{nBatch, q, q}, q is the dimension of decision variable x
                S^{nBatch, q, q} is the set of all positive semi-definite matrices,
                F \in R^{nBatch, q}
                A \in R^{nBatch, n, q},  n is the number of constraint
                b \in R^{nBatch, n}

    Parameters:
            H:  A (nBatch, q, q)  Tensor.
            F:  A (nBatch, q) Tensor.
            A:  A (nBatch, n, q) Tensor.
            b:  A (nBatch, n) Tensor.

    barrier: A (nBatch, n) Tensor

    Returns: x*: a (nBatch, q) Tensor.                       
    
    '''

    _, nConstraint = b.size()

    if nConstraint < 2:
        raise RuntimeError('The number of constraints should be no less than two')
            
    elif nConstraint > 2:
        if option == 'min':
            _, nBarrier = barrier.size()
            if nBarrier != nConstraint:
               raise RuntimeError('The number of constraints should be equal to the number of barrier')

            _, indx = torch.topk(barrier, 2, largest=False)
            G1 = A[:,indx[0],:]
            G2 = A[:,indx[1],:]
            h1 = b[:,indx[0]:indx[0]+1]
            h2 = b[:,indx[1]:indx[1]+1]

        else:
            raise NotImplementedError('The log approach is not yet implemented')    
    else:
        G1 = A[:,0,:]
        G2 = A[:,1,:]
        h1 = b[:,0:1]
        h2 = b[:,1:2]

    
    y1_bar = 1*G1  # H = identity matrix
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


def eBQP_g(H, F, A, b, barrier = [], option = 'min',  check_H_spd=True):

    ''' solve a batch of QP in closed-form, general case
    The QP for each instance in the batch is of the form

                x* =   argmin_x 1/2 x^T H x + F^T x
                        subject to Ax <= b

            where H \in S^{nBatch, q, q}, q is the dimension of decision variable x
                S^{nBatch, q, q} is the set of all positive semi-definite matrices,
                F \in R^{nBatch, q}
                A \in R^{nBatch, n, q},  n is the number of constraint
                b \in R^{nBatch, n}

    Parameters:
            H:  A (nBatch, q, q)  Tensor.
            F:  A (nBatch, q) Tensor.
            A:  A (nBatch, n, q) Tensor.
            b:  A (nBatch, n) Tensor.

    barrier: A (nBatch, n) Tensor

    Returns: x*: a (nBatch, q) Tensor.                       
    
    '''

    if check_H_spd:
        try:
            torch.linalg.cholesky(H)
        except:
            raise RuntimeError('H is not Semi-Positive Definite')

    _, nConstraint = b.size()

    if nConstraint < 2:
        raise RuntimeError('The number of constraints should be no less than two')
            
    elif nConstraint > 2:
        if option == 'min':
            _, nBarrier = barrier.size()
            if nBarrier != nConstraint:
               raise RuntimeError('The number of constraints should be equal to the number of barrier')

            _, indx = torch.topk(barrier, 2, largest=False)
            G1 = A[:,indx[0],:]
            G2 = A[:,indx[1],:]
            h1 = b[:,indx[0]:indx[0]+1]
            h2 = b[:,indx[1]:indx[1]+1]

        else:
            raise NotImplementedError('The log approach is not yet implemented')    
    else:
        G1 = A[:,0,:]
        G2 = A[:,1,:]
        h1 = b[:,0:1]
        h2 = b[:,1:2]

    
    y1_bar = torch.bmm(torch.inverse(H),G1.unsqueeze(2)).squeeze(2) 
    y2_bar = torch.bmm(torch.inverse(H),G2.unsqueeze(2)).squeeze(2)
    u_bar = -torch.bmm(torch.inverse(H),F.unsqueeze(2)).squeeze(2)     # reference control
    p1_bar = h1 - torch.sum(G1*u_bar,dim = 1).unsqueeze(1)
    p2_bar = h2 - torch.sum(G2*u_bar,dim = 1).unsqueeze(1)

    torch.bmm(torch.bmm(y1_bar.unsqueeze(1), H), y1_bar.unsqueeze(2))

    G = torch.cat([torch.bmm(torch.bmm(y1_bar.unsqueeze(1), H), y1_bar.unsqueeze(2)), torch.bmm(torch.bmm(y1_bar.unsqueeze(1), H), y2_bar.unsqueeze(2)), torch.bmm(torch.bmm(y2_bar.unsqueeze(1), H), y1_bar.unsqueeze(2)), torch.bmm(torch.bmm(y2_bar.unsqueeze(1), H), y2_bar.unsqueeze(2))], dim = 2)
    #G = 1*[y1_bar*y1_bar', y1_bar*y2_bar'; y2_bar*y1_bar', y2_bar*y2_bar']
    w_p1_bar = torch.clamp(p1_bar, max=0)
    w_p2_bar = torch.clamp(p2_bar, max=0)

    # G 0-(1,1), 1-(1,2), 2-(2,1), 3-(2,2)
    lambda1 = torch.where(G[:,:,2]*w_p2_bar < G[:,:,3]*p1_bar, torch.zeros_like(p1_bar), torch.where(G[:,:,1]*w_p1_bar < G[:,:,0]*p2_bar, w_p1_bar/G[:,:,0], torch.clamp(G[:,:,3]*p1_bar - G[:,:,2]*p2_bar, max=0)/(G[:,:,0]*G[:,:,3] - G[:,:,1]*G[:,:,2])))
    
    lambda2 = torch.where(G[:,:,2]*w_p2_bar < G[:,:,3]*p1_bar, w_p2_bar/G[:,:,3], torch.where(G[:,:,1]*w_p1_bar < G[:,:,0]*p2_bar, torch.zeros_like(p1_bar), torch.clamp(G[:,:,0]*p2_bar - G[:,:,1]*p1_bar, max=0)/(G[:,:,0]*G[:,:,3] - G[:,:,1]*G[:,:,2])))

    x = lambda1*y1_bar + lambda2*y2_bar + u_bar

    return x