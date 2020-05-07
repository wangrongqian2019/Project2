# project1:TV regularization.py

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
node_name = MPI.Get_processor_name() # get the name of the node

nx=200
ny=200
nz=200


#construct nimg at rank0 and send message
if rank==0:
    img=100.0*np.ones((nx,ny,nz))
    img[75:150,75:150,75:150]=150.0
    nmean,nsigma=0.0,12.0
    nimg=np.random.normal(nmean,nsigma,(nx,ny,nz))+img

## surround nimg with 0. 200x200x200-->202x202x202 
    u=np.zeros((nx+2,ny+2,nz+2))
    u[1:201,1:201,1:201]=nimg   

##send and recv 
    for i in range(1,size):
        comm.send(u[(nx/size*i):(nx/size*(i+1)+2),0:(ny+2),0:(nz+2)],dest=i,tag=11)
        print(i)
        
else:
    u=comm.recv(source=0,tag=11)
    

f=u[1:(nx/size+1),1:(ny+1),1:(nz+1)]


#solving problem by iterative 


alpha=0.01
beta=0.1
gamma=0.01

##initial values mx my mz mux muy muz
mx=np.zeros((nx/size+2,ny+2,nz+2))
my=np.zeros((nx/size+2,ny+2,nz+2))
mz=np.zeros((nx/size+2,ny+2,nz+2))
mux=np.zeros((nx/size+2,ny+2,nz+2))
muy=np.zeros((nx/size+2,ny+2,nz+2))
muz=np.zeros((nx/size+2,ny+2,nz+2))


dt=1

for iter in range(0,10):


##u subproblem
    u_xx=u[2:(nx/size+2),1:(ny+1),1:(nz+1)]+u[0:(nx/size+0),1:(ny+1),1:(nz+1)]-2*u[1:(nx/size+1),1:(ny+1),1:(nz+1)]
    u_yy=u[1:(nx/size+1),2:(ny+2),1:(nz+1)]+u[1:(nx/size+1),0:(ny+0),1:(nz+1)]-2*u[1:(nx/size+1),1:(ny+1),1:(nz+1)]
    u_zz=u[1:(nx/size+1),1:(ny+1),2:(nz+2)]+u[1:(nx/size+1),1:(ny+1),0:(nz+0)]-2*u[1:(nx/size+1),1:(ny+1),1:(nz+1)]
    divmu=mux[1:(nx/size+1),1:(ny+1),1:(nz+1)]+muy[1:(nx/size+1),1:(ny+1),1:(nz+1)]+muz[1:(nx/size+1),1:(ny+1),1:(nz+1)]-mux[0:(nx/size+0),1:(ny+1),1:(nz+1)]-muy[1:(nx/size+1),0:(ny+0),1:(nz+1)]-muz[1:(nx/size+1),1:(ny+1),0:(nz+0)]
    divm=mx[1:(nx/size+1),1:(ny+1),1:(nz+1)]+my[1:(nx/size+1),1:(ny+1),1:(nz+1)]+mz[1:(nx/size+1),1:(ny+1),1:(nz+1)]-mx[0:(nx/size+0),1:(ny+1),1:(nz+1)]-my[1:(nx/size+1),0:(ny+0),1:(nz+1)]-mz[1:(nx/size+1),1:(ny+1),0:(nz+0)]
    u[1:(nx/size+1),1:(ny+1),1:(nz+1)]+=dt*(beta*(u_xx+u_yy+u_zz-divm)+divmu-(u[1:(nx/size+1),1:(ny+1),1:(nz+1)]-f))



##m subproblem
    m_xx=u[2:(nx/size+2),1:(ny+1),1:(nz+1)]-u[1:(nx/size+1),1:(ny+1),1:(nz+1)]+1/beta*mux[1:(nx/size+1),1:(ny+1),1:(nz+1)]
    m_yy=u[1:(nx/size+1),2:(ny+2),1:(nz+1)]-u[1:(nx/size+1),1:(ny+1),1:(nz+1)]+1/beta*muy[1:(nx/size+1),1:(ny+1),1:(nz+1)]
    m_zz=u[1:(nx/size+1),1:(ny+1),2:(nz+2)]-u[1:(nx/size+1),1:(ny+1),1:(nz+1)]+1/beta*mux[1:(nx/size+1),1:(ny+1),1:(nz+1)]
    mx[1:(nx/size+1),1:(ny+1),1:(nz+1)]=(2*(m_xx>0)-1)*(abs(m_xx)-alpha/beta)*(abs(m_xx)>alpha/beta)
    my[1:(nx/size+1),1:(ny+1),1:(nz+1)]=(2*(m_yy>0)-1)*(abs(m_yy)-alpha/beta)*(abs(m_yy)>alpha/beta)
    mz[1:(nx/size+1),1:(ny+1),1:(nz+1)]=(2*(m_zz>0)-1)*(abs(m_zz)-alpha/beta)*(abs(m_zz)>alpha/beta)


##mu subproblem
    mux[1:(nx/size+1),1:(ny+1),1:(nz+1)]+=gamma*(mx[1:(nx/size+1),1:(ny+1),1:(nz+1)]-(u[2:(nx/size+2),1:(ny+1),1:(nz+1)])-u[1:(nx/size+1),1:(ny+1),1:(nz+1)])
    muy[1:(nx/size+1),1:(ny+1),1:(nz+1)]+=gamma*(my[1:(nx/size+1),1:(ny+1),1:(nz+1)]-(u[1:(nx/size+1),2:(ny+2),1:(nz+1)])-u[1:(nx/size+1),1:(ny+1),1:(nz+1)])
    muz[1:(nx/size+1),1:(ny+1),1:(nz+1)]+=gamma*(mz[1:(nx/size+1),1:(ny+1),1:(nz+1)]-(u[1:(nx/size+1),1:(ny+1),2:(nz+2)])-u[1:(nx/size+1),1:(ny+1),1:(nz+1)])

comm.Barrier() 


##communication of processes
if rank<size-1:
    comm.send(u[50,1:201,1:201],dest=rank+1,tag=101)
    comm.send(mx[50,1:201,1:201],dest=rank+1,tag=102)
    comm.send(my[50,1:201,1:201],dest=rank+1,tag=103)
    comm.send(mz[50,1:201,1:201],dest=rank+1,tag=104)
    comm.send(mux[50,1:201,1:201],dest=rank+1,tag=105)
    comm.send(muy[50,1:201,1:201],dest=rank+1,tag=106)
    comm.send(muz[50,1:201,1:201],dest=rank+1,tag=107)
if rank>0:
    u[0,1:201,1:201]=comm.recv(source=rank-1,tag=101)
    mx[0,1:201,1:201]=comm.recv(source=rank-1,tag=102)
    my[0,1:201,1:201]=comm.recv(source=rank-1,tag=103)
    mz[0,1:201,1:201]=comm.recv(source=rank-1,tag=104)
    mux[0,1:201,1:201]=comm.recv(source=rank-1,tag=105)
    muy[0,1:201,1:201]=comm.recv(source=rank-1,tag=106)
    muz[0,1:201,1:201]=comm.recv(source=rank-1,tag=107)

comm.Barrier() 

if rank>0:
    comm.send(u[1,1:201,1:201],dest=rank-1,tag=1001)
    comm.send(mx[1,1:201,1:201],dest=rank-1,tag=1002)
    comm.send(my[1,1:201,1:201],dest=rank-1,tag=1003)
    comm.send(mz[1,1:201,1:201],dest=rank-1,tag=1004)
    comm.send(mux[1,1:201,1:201],dest=rank-1,tag=1005)
    comm.send(muy[1,1:201,1:201],dest=rank-1,tag=1006)
    comm.send(muz[1,1:201,1:201],dest=rank-1,tag=1007)
if rank<size-1:
    u[51,1:201,1:201]=comm.recv(source=rank+1,tag=1001)
    mx[51,1:201,1:201]=comm.recv(source=rank+1,tag=1002)
    my[51,1:201,1:201]=comm.recv(source=rank+1,tag=1003)
    mz[51,1:201,1:201]=comm.recv(source=rank+1,tag=1004)
    mux[51,1:201,1:201]=comm.recv(source=rank+1,tag=1005)
    muy[51,1:201,1:201]=comm.recv(source=rank+1,tag=1006)
    muz[51,1:201,1:201]=comm.recv(source=rank+1,tag=1007)

comm.Barrier() 

#send solutions to rank0

if rank>0:
    comm.send(u[1:(nx/size+1),1:(ny+1),1:(nz+1)],dest=0,tag=111)
else:
    solu=np.ones((nx,ny,nz))
    solu[0:(nx/size),0:ny,0:nz]=u[1:(nx/size+1),1:(ny+1),1:(nz+1)]
    for j in range(1,size):
        solu[(nx/size*j):(nx/size*(j+1)),0:ny,0:nz]=comm.recv(source=j,tag=111)

print(solu)






















