import numpy as np
from scipy import sparse

#=================================================================================================================
# Main Mesh Processing Functions
#=================================================================================================================

def refine(p, e, t): # by Stefan Takacs
    """
    Uniformly refine mesh by subdividing all triangles into 4 congruent ones.
    """
    Np = p.shape[0]
    Ne = e.shape[0]
    Nt = t.shape[0]
    
    # All new data structures preserve the number of colums;
    # columns of e and t without special meaning are inherited
    # by children.
    pnew = np.zeros((Np+Ne,    p.shape[1]))
    enew = np.zeros((2*Ne+3*Nt,e.shape[1]),'i')
    tnew = np.zeros((4*Nt,     t.shape[1]),'i')
    
    # New points are
    #  a) old points (with old indices) 
    pnew[:Np,:] = p
    #  b) new points on midpoint of edges Ei (with index Np+Ei)
    pnew[Np:Np+Ne,:] = (p[e[:,0],:] + p[e[:,1],:])/2

    # New edges are
    #  a) in place of old edge Ei between P0 and P1
    #    aa) new edge 2*i   between P0 and midpoint of Ei
    enew[0:2*Ne:2,:] = e[:,:]
    enew[0:2*Ne:2,1] = range(Np,Np+Ne)
    #    ab) new edge 2*i+1 between P1 and midpoint of Ei
    enew[1:2*Ne:2,:] = e[:,:]
    enew[1:2*Ne:2,0] = range(Np,Np+Ne)
    #  b) inside of triangle i with edges E0, E1 and E2
    #    ba) new edge 2*Ne+3*i   between midpoint of E0 and midpoint of E1
    enew[2*Ne  :2*Ne+3*Nt:3,0] = Np+t[:,3+0]
    enew[2*Ne  :2*Ne+3*Nt:3,1] = Np+t[:,3+1]
    #    bb) new edge 2*Ne+3*i+1 between midpoint of E1 and midpoint of E2
    enew[2*Ne+1:2*Ne+3*Nt:3,0] = Np+t[:,3+1]
    enew[2*Ne+1:2*Ne+3*Nt:3,1] = Np+t[:,3+2]
    #    bc) new edge 2*Ne+3*i+2 between midpoint of E2 and midpoint of E0
    enew[2*Ne+2:2*Ne+3*Nt:3,0] = Np+t[:,3+2]
    enew[2*Ne+2:2*Ne+3*Nt:3,1] = Np+t[:,3+0]
    
    # New triangles are in place of old triangle Ti with
    #   a) corners P0, midpoint of E0 and midpoint of E2
    tnew[0:4*Nt:4,:] = t[:,:]
    tnew[0:4*Nt:4,0] = t[:,0]
    tnew[0:4*Nt:4,1] = Np+t[:,3+0]
    tnew[0:4*Nt:4,2] = Np+t[:,3+2]
    tnew[0:4*Nt:4,3] = 2*t[:,3+0] + ( enew[2*t[:,3+0]+1,1] == t[:,0] )
    tnew[0:4*Nt:4,4] = 2*Ne+3*np.arange(Nt)+2
    tnew[0:4*Nt:4,5] = 2*t[:,3+2] + ( enew[2*t[:,3+2]+1,1] == t[:,0] )
    #   b) corners P1, midpoint of E1 and midpoint of E0
    tnew[1:4*Nt:4,:] = t[:,:]
    tnew[1:4*Nt:4,0] = t[:,1]
    tnew[1:4*Nt:4,1] = Np+t[:,3+1]
    tnew[1:4*Nt:4,2] = Np+t[:,3+0]
    tnew[1:4*Nt:4,3] = 2*t[:,3+1] + ( enew[2*t[:,3+1]+1,1] == t[:,1] )
    tnew[1:4*Nt:4,4] = 2*Ne+3*np.arange(Nt)+0
    tnew[1:4*Nt:4,5] = 2*t[:,3+0] + ( enew[2*t[:,3+0]+1,1] == t[:,1] )
    #   c) corners P2, midpoint of E2 and midpoint of E1
    tnew[2:4*Nt:4,:] = t[:,:]
    tnew[2:4*Nt:4,0] = t[:,2]
    tnew[2:4*Nt:4,1] = Np+t[:,3+2]
    tnew[2:4*Nt:4,2] = Np+t[:,3+1]
    tnew[2:4*Nt:4,3] = 2*t[:,3+2] + ( enew[2*t[:,3+2]+1,1] == t[:,2] )
    tnew[2:4*Nt:4,4] = 2*Ne+3*np.arange(Nt)+1
    tnew[2:4*Nt:4,5] = 2*t[:,3+1] + ( enew[2*t[:,3+1]+1,1] == t[:,2] )
    #   d) corners midpoint of E0, midpoint of E1 and midpoint of E2
    tnew[3:4*Nt:4,:] = t[:,:]
    tnew[3:4*Nt:4,0] = Np+t[:,3+0]
    tnew[3:4*Nt:4,1] = Np+t[:,3+1]
    tnew[3:4*Nt:4,2] = Np+t[:,3+2]
    tnew[3:4*Nt:4,3] = 2*Ne+3*np.arange(Nt)+0 ### TODO: set to int or use range
    tnew[3:4*Nt:4,4] = 2*Ne+3*np.arange(Nt)+1
    tnew[3:4*Nt:4,5] = 2*Ne+3*np.arange(Nt)+2
    
    return [pnew, enew, tnew]

def refine_n_times(p, e, t, number_of_refinements: int= 3):
    """
    refines the mesh n times
    """
    for i in range(number_of_refinements):
        p, e, t = refine(p, e, t)
        
    return p, e, t

def fix_orientation(p, tri):
    tri = tri.copy()

    det = (p[tri[:, 1], 0] - p[tri[:, 0], 0]) * (p[tri[:, 2], 1] - p[tri[:, 0], 1]) - \
          (p[tri[:, 2], 0] - p[tri[:, 0], 0]) * (p[tri[:, 1], 1] - p[tri[:, 0], 1])

    flip = det < 0

    tmp = tri[flip, 1].copy()
    tri[flip, 1] = tri[flip, 2]
    tri[flip, 2] = tmp

    return tri

def build_stable_mesh(p, tri_idx):
    tri_idx = fix_orientation(p, tri_idx)

    edge_dict = {}
    edges = []
    tri_edges = []

    for tri in tri_idx:
        local_edges = []

        for i in range(3):
            a = tri[i]
            b = tri[(i + 1) % 3]
            edge = tuple(sorted((a, b)))

            if edge not in edge_dict:
                edge_dict[edge] = len(edges)
                edges.append(edge)

            local_edges.append(edge_dict[edge])

        tri_edges.append(local_edges)

    e = np.zeros((len(edges), 3), dtype=int)
    e[:, :2] = np.array(edges)

    t = np.zeros((tri_idx.shape[0], 7), dtype=int)
    t[:, :3] = tri_idx
    t[:, 3:6] = np.array(tri_edges)

    # Mark boundary edges
    counts = np.zeros(len(edges), dtype=int)
    for tri in t:
        for i in range(3):
            counts[tri[3+i]] += 1

    e[counts == 1, 2] = 1  # boundary flag

    return p, e, t

def embeddingForRefined(p, e, t):  # by Stefan Takacs
     """
     Get embedding of Courant element for mesh into uniformly refined mesh.
     """
     Np = p.shape[0]
     Ne = e.shape[0]
     data = np.zeros( (Np+2*Ne,) )
     data[:Np] = 1.
     data[Np:] = .5

     rowidx = np.arange(Np+2*Ne)
     rowidx[Np:] = np.ravel(np.kron(np.arange(Np,Np+Ne),[1,1]))
     colidx = np.arange(Np+2*Ne)
     colidx[Np:] = np.ravel(e[:,0:2])
     
     embedding = sparse.csc_matrix((np.ravel(data),(rowidx,colidx)),shape=(Np+Ne,Np))
     
     return embedding

