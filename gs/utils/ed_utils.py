import torch
from torch.autograd import Function
from mpmath import *
import numpy as np

## usage
# from utils.ed_utils import Batched_ED #QR-based ED for dim<32
# batched_ed = Batched_ED.apply
# eigen_vectors,eigen_values = batched_ed(cov)

mp.dps = 20
one = mpf(1)
mp.pretty = True
def f(x):
    return one/(one-x)
a = taylor(f, 0, 100)
#[50, 50] Pade coefficients of the geometric series
pade_p, pade_q = pade(a, 50, 50)
pade_p = torch.from_numpy(np.array(pade_p).astype(float)).cuda()
pade_q = torch.from_numpy(np.array(pade_q).astype(float)).cuda()

#stable and close approximation of SVD gradients, see: https://arxiv.org/abs/2105.02498
def pade_approximants(s):
    s = torch.diagonal(s, dim1=1, dim2=2)
    dtype = s.dtype
    I = torch.eye(s.shape[1], device=s.device).type(dtype).view(1,s.shape[1],s.shape[1]).repeat(s.shape[0],1,1)
    p = s.unsqueeze(-1) / s.unsqueeze(-2) - I
    p = torch.where(p < 1., p, 1. / p)
    a1 = s.view(s.shape[0],s.shape[1],1).repeat(1, 1, s.shape[1])#.transpose(1,2)
    a1_t = a1.transpose(1,2)
    a1 = 1. / torch.where(a1 >= a1_t, a1, - a1_t)
    a1 *= torch.ones(s.shape[1], s.shape[1], device=s.device).type(dtype).view(1,s.shape[1],s.shape[1]).repeat(s.shape[0],1,1) - I
    p_app = torch.ones_like(p)*pade_p[0]
    q_app = torch.ones_like(p)*pade_q[0]
    p_hat = torch.ones_like(p)
    for i in range(50):
        p_hat = p_hat * p
        p_app += pade_p[i+1]*p_hat
        q_app += pade_q[i+1]*p_hat
    a1 = a1 * p_app / q_app #rational approximation
    return a1

#close approximation of SVD gradients, see: https://arxiv.org/abs/2105.02498
def taylor_polynomial(s):
    s = torch.diagonal(s, dim1=1, dim2=2)
    dtype = s.dtype
    I = torch.eye(s.shape[1], device=s.device).type(dtype).view(1,s.shape[1],s.shape[1]).repeat(s.shape[0],1,1)
    p = s.unsqueeze(-1) / s.unsqueeze(-2) - I
    p = torch.where(p < 1., p, 1. / p)
    a1 = s.view(s.shape[0],s.shape[1],1).repeat(1, 1, s.shape[1])
    a1_t = a1.transpose(1,2)
    a1 = 1. / torch.where(a1 >= a1_t, a1, - a1_t)
    a1 *= torch.ones(s.shape[1], s.shape[1], device=s.device).type(dtype).view(1,s.shape[1],s.shape[1]).repeat(s.shape[0],1,1) - I
    p_app = torch.ones_like(p)
    p_hat = torch.ones_like(p)
    for i in range(100):
        p_hat = p_hat * p
        p_app += p_hat
    a1 = a1 * p_app
    return a1

class Batched_ED(Function):
    @staticmethod
    def forward(ctx, input):
        cov = input
        batch_size, input_dim,_ = input.size()
        # Householder reflector to tri-diagonal form
        tri_cov, tri_vector = Householder_reflection(cov,batch_size,input_dim)
        # Givens rotation to diagonal form
        eig_diag, diag_vector = QR_Givens_Shift(tri_cov,batch_size,input_dim,input_dim)
        eig_diag = torch.diag_embed(torch.clamp(torch.diagonal(eig_diag,dim1=1,dim2=2),min=1e-10))
        eig_vec = tri_vector.bmm(diag_vector)
        ctx.save_for_backward(eig_vec, eig_diag)
        return eig_vec,eig_diag

    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        eig_vec, eig_diag = ctx.saved_tensors
        eig_vec_deri, eig_diag_deri = grad_output1, grad_output2
        k = taylor_polynomial(eig_diag)
        # Gradient Overflow Check
        k[k == float('inf')] = k[k != float('inf')].max()
        k[k == float('-inf')] = k[k != float('-inf')].min()
        k[k != k] = k.max()
        grad_input = (k.transpose(1, 2) * (eig_vec.transpose(1, 2).bmm(eig_vec_deri))) + torch.diag_embed(
            torch.diagonal(eig_diag_deri, dim1=1, dim2=2))
        # Gradient Overflow Check
        grad_input[grad_input == float('inf')] = grad_input[grad_input != float('inf')].max()
        grad_input[grad_input == float('-inf')] = grad_input[grad_input != float('-inf')].min()
        grad_input = eig_vec.bmm(grad_input).bmm(eig_vec.transpose(1, 2))
        # Gradient Overflow Check
        grad_input[grad_input == float('inf')] = grad_input[grad_input != float('inf')].max()
        grad_input[grad_input == float('-inf')] = grad_input[grad_input != float('-inf')].min()
        return grad_input

#Return 2 eigenvalues of the 2x2 right-bottom block
def wilkinson_shift(matrix):
  eps=1e-10
  sigma = (matrix[:,0,0]-matrix[:,1,1])/2
  temp = (torch.sign(sigma) * matrix[:,0,1]**2) / (abs(sigma)+torch.sqrt(sigma**2+matrix[:,0,1]**2)+eps)
  return matrix[:,0,0]-temp, matrix[:,1,1] - temp

def house_two_sides(A,mu):
  eps = 1e-10
  H = 0.5 * (torch.norm(mu+eps,dim=1).type(A.dtype).view(mu.size(0),1,1))**2
  p = A.bmm(mu)/H
  K = mu.transpose(1,2).bmm(p)/2/H
  q = p - K*(mu)
  return A - q.bmm(mu.transpose(1,2)) - mu.bmm(q.transpose(1,2))

#Givens Rotation based QR iteration with shifts
def QR_Givens_Shift(cov,batch_size,input_dim,iterations):
  eps = 1e-10
  Givens = torch.eye(input_dim,device=cov.device).view(1,input_dim,input_dim,1).repeat(batch_size,1,1,input_dim-1)
  shift = torch.eye(input_dim,device=cov.device).view(1,input_dim,input_dim).repeat(batch_size,1,1)
  reduction=0
  Givens_single = shift.clone()
  Givens_all = shift.clone()
  for iteration in range(iterations):
    shift_coe1, shift_coef2  = wilkinson_shift(cov[:,input_dim-2-reduction:input_dim-reduction,input_dim-2-reduction:input_dim-reduction])
    ## 1st Shift
    # 1st Rotation
    tmp_shift = shift_coef2.view(shift.size(0),1,1)*shift
    cov  =  cov - tmp_shift
    Givens[:,0:2,0,0] = cov[:,0,0:2] / (torch.sqrt(torch.sum(cov[:,0,0:2]**2,dim=1)).view(cov.size(0),1)+eps)
    Givens[:,1,1,0] = Givens[:,0,0,0]
    Givens[:,0,1,0] = - Givens[:,1,0,0]
    cov[:,0:3,0:3]=Givens[:,0:3,0:3,0].transpose(1, 2).bmm(cov[:,0:3,0:3]).bmm(Givens[:,0:3,0:3,0])
    Givens_single[:,0:3,0:3] = Givens_single[:,0:3,0:3].bmm(Givens[:,0:3,0:3,0])
    for i in range(1,input_dim-1-reduction):
      Givens[:,i:i+2,i,i] = cov[:,i,i:i+2] / (torch.sqrt(torch.sum(cov[:,i,i:i+2]**2,dim=1)).view(cov.size(0),1)+eps)
      Givens[:,i+1,i+1,i] = Givens[:,i,i,i]
      Givens[:,i,i+1,i] = - Givens[:,i+1,i,i]
      cov[:,i-1:i+3,i-1:i+3] = Givens[:,i-1:i+3,i-1:i+3,i].transpose(1, 2).bmm(cov[:,i-1:i+3,i-1:i+3]).bmm(Givens[:,i-1:i+3,i-1:i+3,i])
      Givens_single[:,:,i:] = Givens_single[:,:,i:].bmm(Givens[:,i:,i:,i])
    Givens_all = Givens_all.bmm(Givens_single)
    Givens_single = shift.clone()
    cov = cov + tmp_shift
    if abs(cov[:,input_dim-1-reduction,input_dim-2-reduction]).max()<1e-3: #Perform Shrinkage
      reduction = reduction + 1
    if reduction>=input_dim-2:
      break
    ### 2nd Shift
    tmp_shift = shift_coe1.view(shift.size(0),1,1)*shift
    cov  =  cov - tmp_shift
    Givens[:,0:2,0,0] = cov[:,0,0:2] / (torch.sqrt(torch.sum(cov[:,0,0:2]**2,dim=1)).view(cov.size(0),1)+eps)
    Givens[:,1,1,0] = Givens[:,0,0,0]
    Givens[:,0,1,0] = - Givens[:,1,0,0]
    cov[:,0:3,0:3]=Givens[:,0:3,0:3,0].transpose(1, 2).bmm(cov[:,0:3,0:3]).bmm(Givens[:,0:3,0:3,0])
    Givens_single[:,0:3,0:3] = Givens_single[:,0:3,0:3].bmm(Givens[:,0:3,0:3,0])
    for i in range(1,input_dim-1-reduction):
      Givens[:,i:i+2,i,i] = cov[:,i,i:i+2] / (torch.sqrt(torch.sum(cov[:,i,i:i+2]**2,dim=1)).view(cov.size(0),1)+eps)
      Givens[:,i+1,i+1,i] = Givens[:,i,i,i]
      Givens[:,i,i+1,i] = - Givens[:,i+1,i,i]
      cov[:,i-1:i+3,i-1:i+3] = Givens[:,i-1:i+3,i-1:i+3,i].transpose(1, 2).bmm(cov[:,i-1:i+3,i-1:i+3]).bmm(Givens[:,i-1:i+3,i-1:i+3,i])
      Givens_single[:,:,i:] = Givens_single[:,:,i:].bmm(Givens[:,i:,i:,i])
    Givens_all = Givens_all.bmm(Givens_single)
    Givens_single = shift.clone()
    cov = cov + tmp_shift
    if abs(cov[:,input_dim-1-reduction,input_dim-2-reduction]).max()<1e-3: #Perform Shrinkage
      reduction = reduction + 1
    if reduction>=input_dim-2:
      break
  return cov, Givens_all

def Householder_reflection(cov, batch_size, input_dim):
    eps=1e-10
    hv = torch.zeros(batch_size, input_dim, input_dim - 2, requires_grad=False, device=cov.device)
    eigen_vectors = torch.eye(input_dim, requires_grad=False, device=cov.device).view(1, input_dim, input_dim).repeat(
        batch_size, 1, 1)
    eye_matrix = torch.eye(input_dim, requires_grad=False, device=cov.device).view(1, input_dim, input_dim).repeat(
        batch_size, 1, 1)
    for i in range(2, input_dim):
        cov_temp = cov.clone()
        hv[:, i - 1:input_dim, i - 2] = cov_temp[:, i - 2, i - 1:input_dim]
        hv[:, i - 1, i - 2] = hv[:, i - 1, i - 2] + (hv[:, i - 1, i - 2] / (abs(hv[:, i - 1, i - 2])+eps))* torch.sqrt(
            torch.sum(hv[:, i - 1:input_dim, i - 2] ** 2, dim=1))
        hm = eye_matrix - 2 * torch.bmm(hv[:, :, i - 2:i - 1], (hv[:, :, i - 2:i - 1].transpose(1, 2))) / (
            torch.norm(hv[:, :, i - 2]+eps, dim=1).view(batch_size, 1, 1)) ** 2
        eigen_vectors = eigen_vectors.bmm(hm)
        cov = house_two_sides(cov, hv[:, :, i - 2:i - 1])
    return cov, eigen_vectors