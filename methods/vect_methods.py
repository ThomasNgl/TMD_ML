import numpy as np

from gudhi.representations.vector_methods import Atol, ComplexPolynomial, BettiCurve, Landscape, Silhouette
from methods.utils import tent_functions, define_limits, pi_function, sort_bars
from sklearn.cluster import KMeans

#Vectorizations
# list_phs is a list of phs
# A ph is np.array of shape (nb_bars, 2).
# It is the same thing as X in the method compute_phs_from_pop_list()
#0
def hist_vectorization(list_phs, resolution = 100, is_space =True):
    matrix = []
    if is_space:
        xlim, ylim = define_limits(list_phs)
        space = np.linspace(int(np.min((xlim, ylim))), int(np.max((xlim, ylim))), resolution)
    
    for ph in list_phs:
        vector = []
        if not is_space:
            space = np.linspace(int(np.min(ph)), int(np.max(ph)), resolution)
        for t in space:
            alive = 0
            for i in range(len(ph)):
                if (ph[i][1] < t and ph[i][0] > t) or (ph[i][1] > t and ph[i][0] < t):
                    alive += 1
            vector.append(alive)
        matrix.append(np.array(vector)/sum(vector)*len(ph))
    return np.array(matrix)
#1
def entropy(list_phs, resolution = None, is_space =True):
    matrix = []
    #If resolution is none, Entropy is a real value for each ph
    if not resolution:
        for ph in list_phs:
            #Compute L as in the survey 
            L = 0
            for i in range(len(ph)):
                L += abs(ph[i][1]-ph[i][0])
            
            #Compute entropy following the formula
            #In TMD case, death can be < birth, thus compute abs of death - birth
            ent = 0
            for i in range(len(ph)):
                if 0 != abs(ph[i][1]-ph[i][0]):
                    ent += abs(ph[i][1]-ph[i][0])/L * np.log(abs(ph[i][1]-ph[i][0])/L)
                #Sometimes death = birth, weird???, so don't count them
                else:
                    print('lifetime = 0 for bar ', i, ' in a barcode ')
            matrix.append(-ent)
    
    #Entropy is a vector of size resolution for each ph
    else: 
        if is_space:
            xlim, ylim =define_limits(list_phs)
            space = np.linspace(int(np.min((xlim, ylim))), int(np.max((xlim, ylim))), resolution)
        
        for ph in list_phs:
            #Compute L as in the survey 
            L = 0
            for i in range(len(ph)):
                L += abs(ph[i][1]-ph[i][0])
            
            #Compute entropy following the formula
            #In TMD case death can be < birth, thus compute abs of death - birth
            vector = []
            if not is_space:
                space = np.linspace(int(np.min(ph)), int(np.max(ph)), resolution)
            for t in space:
                ent = 0
                for i in range(len(ph)):
                    if (ph[i][1] < t and ph[i][0] > t) or (ph[i][1] > t and ph[i][0] < t) and 0 != abs(ph[i][1]-ph[i][0]):
                        ent += abs(ph[i][1]-ph[i][0])/L * np.log(abs(ph[i][1]-ph[i][0])/L)
                vector.append(-ent)
            matrix.append(vector)
    return np.array(matrix)
#2
def stats_vectorization(list_phs, features =['mean_std', 'median', 'interquart', 'range', 'q10', 'q25', 'q75', 'q90', 'nbbars', 'entropy', 'entropy_curve']):
    matrix = []
    for ph in list_phs:
        vector = []

        if 'mean_std' in features:
            mean_birth, mean_death = np.mean(ph, 0)
            mean_mid = np.mean(np.mean(ph, 1))
            mean_life = np.mean(abs(ph[:,1]-ph[:,0]))
            
            vector.append(mean_birth)
            vector.append(mean_death) 
            vector.append(mean_mid)
            vector.append(mean_life)

            std_birth, std_death  = np.std(ph, 0)
            std_mid = np.std(np.mean(ph, 1))
            std_life = np.std(abs(ph[:,1]-ph[:,0]))
            
            vector.append(std_birth)
            vector.append(std_death)
            vector.append(std_mid)
            vector.append(std_life)

        if 'median' in features:
            median_birth, median_death = np.median(ph, 0)
            median_mid = np.median(np.mean(ph, 1))
            median_life = np.median(abs(ph[:,1]-ph[:,0]))
            
            vector.append(median_birth)
            vector.append(median_death)
            vector.append(median_mid)
            vector.append(median_life)

        if 'interquart' in features:
            inter_birth, inter_death = np.quantile(ph, 0.75, axis = 0)-np.quantile(ph, 0.25, axis = 0)
            inter_mid = np.quantile(np.mean(ph, 1), 0.75) - np.quantile(np.mean(ph, 1), 0.25)
            inter_life = np.quantile(abs(ph[:,1]-ph[:,0]), 0.75) - np.quantile(abs(ph[:,1]-ph[:,0]), 0.25)
            
            vector.append(inter_birth)
            vector.append(inter_death)
            vector.append(inter_mid)
            vector.append(inter_life)

        if 'range' in features:
            range_birth, range_death = np.quantile(ph, 1., axis = 0) - np.quantile(ph, 0., axis = 0) 
            range_mid = np.quantile(np.mean(ph, 1), 1.) - np.quantile(np.mean(ph, 1), 0.) 
            range_life = np.quantile(abs(ph[:,1]-ph[:,0]), 1.) - np.quantile(abs(ph[:,1]-ph[:,0]), 0.) 
            
            vector.append(range_birth)
            vector.append(range_death)
            vector.append(range_mid)
            vector.append(range_life)
        
        if 'q10' in features:
            q10_birth, q10_death = np.quantile(ph, 0.1, axis = 0) 
            q10_mid = np.quantile(np.mean(ph, 1), 0.1) 
            q10_life = np.quantile(abs(ph[:,1]-ph[:,0]), 0.1) 
            
            vector.append(q10_birth)
            vector.append(q10_death)
            vector.append(q10_mid)
            vector.append(q10_life)

        if 'q25' in features:
            q25_birth, q25_death = np.quantile(ph, 0.25, axis = 0) 
            q25_mid = np.quantile(np.mean(ph, 1), 0.25) 
            q25_life = np.quantile(abs(ph[:,1]-ph[:,0]), 0.25) 
            
            vector.append(q25_birth)
            vector.append(q25_death)
            vector.append(q25_mid)
            vector.append(q25_life)

        if 'q75' in features:
            q75_birth, q75_death = np.quantile(ph, 0.75, axis = 0) 
            q75_mid = np.quantile(np.mean(ph, 1), 0.75) 
            q75_life = np.quantile(abs(ph[:,1]-ph[:,0]), 0.75) 
            
            vector.append(q75_birth)
            vector.append(q75_death)
            vector.append(q75_mid)
            vector.append(q75_life)

        if 'q90' in features:
            q90_birth, q90_death = np.quantile(ph, 0.9, axis = 0) 
            q90_mid = np.quantile(np.mean(ph, 1), 0.9) 
            q90_life = np.quantile(abs(ph[:,1]-ph[:,0]), 0.9) 
            
            vector.append(q90_birth)
            vector.append(q90_death)
            vector.append(q90_mid)
            vector.append(q90_life)

        if 'nbbars' in features:
            vector.append(len(ph))

        if 'entropy' in features:
            vector.append(entropy([ph])[0])

        if 'entropy_curve' in features:
            vect = entropy([ph], resolution = 100)[0]
            for v in vect:
                vector.append(v)
        matrix.append(vector)
    return np.array(matrix)
#3
def algebraic_functions(list_phs, nb_functions = 4):
    matrix = []
    for ph in list_phs:
        lifetimes = abs(ph[:,0]-ph[:,1])
        maximum = np.max(ph)
        vector = []
        #for k in range(nb_functions):
        f1 = 0
        for bar, life in zip(ph, lifetimes):
            f1 += bar[0] * life
        vector.append(f1)

        f2 = 0
        for bar, life in zip(ph, lifetimes):
            f2 += (maximum-bar[1]) * life
        vector.append(f2)

        f3 = 0
        for bar, life in zip(ph, lifetimes):
            f3 += bar[0]**2 * life**4
        vector.append(f3)

        f4 = 0
        for bar, life in zip(ph, lifetimes):
            f4 += (maximum-bar[1])**2 * life**4
        vector.append(f4)

        matrix.append(vector)

    return np.array(matrix)
#4
def tropical_coordinate_functions(list_phs, nb_functions = 7, r = 1):
    matrix = []
    for ph in list_phs:
        lifetimes = abs(ph[:,0]-ph[:,1])
        lifetimes_sorted = np.sort(lifetimes)
        nb_bars = len(ph)
        vector = []
        #for k in range(nb_functions):
        f1 = lifetimes_sorted[-1]
        vector.append(f1)

        f2 = lifetimes_sorted[-1] + lifetimes_sorted[-2]  
        vector.append(f2)

        f3 = lifetimes_sorted[-1] + lifetimes_sorted[-2] + lifetimes_sorted[-3]
        vector.append(f3)

        f4 = lifetimes_sorted[-1] + lifetimes_sorted[-2] + lifetimes_sorted[-3] + lifetimes_sorted[-4]
        vector.append(f4)

        f5 = np.sum(lifetimes)
        vector.append(f5)

        f6 = 0
        for i in range(nb_bars):
            f6 += min(r*lifetimes[i], ph[i][0], ph[i][1])
        vector.append(f6)
       
        mins = []
        for i in range(nb_bars):
            mins.append(min(r*lifetimes[i], ph[i][0], ph[i][1])+lifetimes[i])
        f7 = nb_bars*max(mins) - np.sum(mins)
        vector.append(f7)

        matrix.append(vector)
    return np.array(matrix)
#5
def complex_polynomials(list_phs, sort = False, polynomial_type='R', threshold=10):
    #plot complex polynomials
    if sort:
        list_phs = sort_bars(list_phs)
    zpoly = ComplexPolynomial(polynomial_type=polynomial_type, threshold=threshold)
    matrix = []
    for ph in list_phs:
        vector = zpoly(ph)
        r_vector = []
        for z in vector:
            r_vector.append(z.real)
            r_vector.append(z.imag)
        matrix.append(r_vector)
    return np.array(matrix)
#6   
def betti_curve(list_phs, sort = False, resolution=100, sample_range=[np.nan, np.nan], predefined_grid=None, keep_endpoints=False):
    betti_curv = BettiCurve(resolution=resolution, sample_range=sample_range, predefined_grid=predefined_grid, keep_endpoints=keep_endpoints)
    if sort:
        list_phs = sort_bars(list_phs)
    matrix = betti_curv.fit_transform(list_phs)
    
    return np.array(matrix)        
#7  
def lifespan_curve(list_phs, sort = False, resolution = 100, is_space =True):
    #Sort bars such that birth < death always
    if sort:
        list_phs = sort_bars(list_phs)
    matrix = []
    if is_space:
        xlim, ylim =define_limits(list_phs)
        space = np.linspace(int(np.min((xlim, ylim))), int(np.max((xlim, ylim))), resolution)
    
    for ph in list_phs:
        vector = []
        if not is_space:
            space = np.linspace(int(np.min(ph)), int(np.max(ph)), resolution)
        for t in space:
            life = 0
            for i in range(len(ph)):
                if (ph[i][1] < t and ph[i][0] > t) or (ph[i][1] > t and ph[i][0] < t):
                    life += ph[i][1]-ph[i][0]
            vector.append(life)
        matrix.append(vector)
    return np.array(matrix)
#8   
def landscape(list_phs, num_landscapes=5, resolution=100, sample_range=[np.nan, np.nan], keep_endpoints=False):
    #Try to keep big num landscapes 
    list_phs = sort_bars(list_phs)
    land = Landscape(num_landscapes=num_landscapes, resolution=resolution, sample_range=sample_range, keep_endpoints=keep_endpoints)
    matrix = land.fit_transform(list_phs)
    return np.array(matrix)
#9
def silhouette(list_phs, weight= lambda x:1,
               resolution=100, sample_range=[np.nan, np.nan], keep_endpoints=False):
    list_phs = sort_bars(list_phs)
    sil = Silhouette(weight= weight,
               resolution=resolution, sample_range=sample_range, keep_endpoints=keep_endpoints)
    matrix = sil.fit_transform(list_phs)
    return np.array(matrix)
#10
def PI(list_phs, resolution = 100, norm = True , prop = True, std = None, flatten = True):
    xlim, ylim = define_limits(list_phs)
    if flatten:
        X_pi = [pi_function(ph, xlim = xlim, ylim = ylim, std = std, resolution=resolution).flatten() for ph in list_phs]
    else:
        X_pi = [pi_function(ph, xlim = xlim, ylim = ylim, std = std, resolution=resolution) for ph in list_phs]

    if norm and prop:
        X_pi = [X_pi[i]/np.sum(X_pi[i])*len(list_phs[i]) for i in range(len(list_phs))]
    elif norm:
        X_pi = [X_pi[i]/np.sum(X_pi[i]) for i in range(len(list_phs))]
    elif prop:
        X_pi = [X_pi[i]*len(list_phs[i]) for i in range(len(list_phs))]
    else:
        X_pi = [X_pi[i]/np.max(X_pi[i]) for i in range(len(list_phs))]
    
    return np.array(X_pi)
#11
def template_function(list_phs, resolution = 100, delta = 1):
#https://github.com/lucho8908/adaptive_template_systems
    list_phs = sort_bars(list_phs)
    xlim, ylim = define_limits(list_phs)

    matrix = []
    for ph in list_phs:
        vector = []
        for i in np.linspace(xlim[0], xlim[1], resolution):
            for j in np.linspace(ylim[0]-xlim[0], ylim[1]-xlim[1], resolution):
                tent = tent_functions(ph, [i,j], delta=delta)
                vector.append(tent)
        matrix.append(vector)
    return np.array(matrix)
#12 
#def Adaptive_template_system()
#13
def ATOL(list_phs, quantizer = KMeans(n_clusters=2), weighting_method = 'cloud', contrast = 'gaussian'):
    atol = Atol(quantiser=quantizer, weighting_method= weighting_method, contrast =contrast)
    matrix = atol.fit_transform(list_phs)
    return np.array(matrix)
#14
# def signature(list_phs):
    #gudhi TopologicalVector
#15
def ratio(list_phs, resolution = 100, is_space =True):
    #Sort bars such that birth < death always
    list_phs = sort_bars(list_phs)
    matrix = []
    if is_space:
        xlim, ylim =define_limits(list_phs)
        space = np.linspace(int(np.min((xlim, ylim))), int(np.max((xlim, ylim))), resolution)
    
    for ph in list_phs:
        vector = []
        if not is_space:
            space = np.linspace(int(np.min(ph)), int(np.max(ph)), resolution)
        for t in space:
            life = 0
            for i in range(len(ph)):
                if ((ph[i][1] < t and ph[i][0] > t) or (ph[i][1] > t and ph[i][0] < t)):
                    if ph[i][1] !=0:
                        life += ph[i][0]/(ph[i][1])
                    else:
                        life += ph[i][0]/(ph[i][1]+0.01)
                
            vector.append(life)
        matrix.append(vector)
    return np.array(matrix)
##############