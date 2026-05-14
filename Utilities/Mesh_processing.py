import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri

#=================================================================================================================

def refine(p, e, t):
    """Uniformly refine mesh by subdividing all triangles into 4 congruent ones."""
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

def mesh_df(p, e, t, 
            first_n_entries: int = 9):

    print(f"p is of shape: {p.shape}\ne is of shape: {e.shape}\nt is of shape: {t.shape}")

    df_p = pd.DataFrame(p[:first_n_entries,:], columns=["x1", "x2"])
    df_e = pd.DataFrame(e[:first_n_entries,:], columns=["Node1", "Node2", "Flag"])
    df_t = pd.DataFrame(t[:first_n_entries,:], columns=["V1", "V2", "V3", "E1", "E2", "E3", "Sub"])

    df = pd.concat([df_p, df_e, df_t], axis=1, keys=['p', 'e', 't'])

    def highlight_by_category(col):
        category = col.name[0] 
        
        if category == 'p':
            return ["background-color: #73BA40; color: black"] * len(col) # Blue
        elif category == 'e':
            return ["background-color: #96D44A; color: black"] * len(col) # Green
        elif category == 't':
            return ["background-color: #34623F; color: white"] * len(col) # Red
        return [""] * len(col)

    return (
        df.style
          .apply(highlight_by_category, axis=0)
          .hide(axis="index")
          .set_table_styles([
              {'selector': 'th', 'props': [('text-align', 'center'), 
                                           ('border', '1px solid #ddd'),
                                           ('padding', '8px')]}
          ])
          .format(precision=4)
    )

def Plot_Initial_Refined_meshes(data_path: str, num_of_refinements: int = 3,
                                plot: bool=True,
                                figsize: tuple=(16,8)):
    """
    Plots the initial blender mesh and the refined counterpart. 
    Additionally outputs the refined mesh arrays.
    """
    
    data = np.load(data_path)
    p_raw = data['p']
    tri_idx = data['t_raw']
    data.close()

    p, e, t = build_stable_mesh(p_raw, tri_idx)
    p, e, t = refine_n_times(p, e, t, number_of_refinements=num_of_refinements)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    ax[0].set_title('Initial Mesh')
    ax[1].set_title('Refined Mesh')

    ax[0].triplot(p_raw[:, 0], p_raw[:, 1], tri_idx, color='blue', lw=1, label='Edges')
    ax[0].plot(p_raw[:, 0], p_raw[:, 1], 'ro', markersize=3, label='Nodes')

    ax[1].triplot(p[:, 0], p[:, 1], t[:, :3], color='blue', lw=0.3, label='Edges')
    ax[1].plot(p[:, 0], p[:, 1], 'ro', markersize=1, label='Nodes')

    for i in [0,1]:
        x_min, x_max = ax[i].get_xlim()
        y_min, y_max = ax[i].get_ylim()

        margin_x = np.abs(x_max-x_min)*0.05
        margin_y = np.abs(y_max-y_min)*0.05

        ax[i].set_xlim(x_min - margin_x, x_max + margin_x)
        ax[i].set_ylim(y_min - margin_y, y_max + margin_y)
        ax[i].set_aspect('equal')
        ax[i].legend()
 
    plt.suptitle(f'Initial Mesh ({len(p_raw)} Nodes, {len(tri_idx)} Triangles) --> Refined Mesh ({len(p)} Nodes, {len(t)} Triangles)')

    if plot==True:
        plt.show()
    else:
        plt.close()

    return (p, e, t)