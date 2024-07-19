#RQ : pour exécuter le code, il est nécessaire d'enlever tous les typages des paramètres d'entrée et de sortie... Sans quoi, cela ne fonctionne pas !!! 


#1/Environnement
"""
    a) La classe Regle qui définit les règles

    b)La classe Joueur qui définit ce qu'est un joueur.
        i/ la class MCTS : joueur jouant selon la stratégie MCTS
        ii/ la class Montecarlo : joueur jouant selon la méthode Montecarlo.
        iii/ la class Aléatoire : joueur Random.
        
        tous les joueurs ont la même manière de commencer la partie lorsqu'ils sont J1. 
        Cela permet de les "mettre sur un pied d'égalité".

    c) la classe Arbre de jeu, qui définit les fonctions de travail sur un arbre de jeu...

    d) La classe Partie, qui fait tourner une partie et définit les éléments qui 
    permettent de définir l'état d'une partie.
"""


#2/Fonctions effets de bords
"""
    Les fonctions tour pour un joueur, les fonctions de la classe partie."""


#3/ Fonctions calculatrices sans effet de bord.
""" Les fonctions qui doivent calculer l'état de la partie.
    Par ailleurs, on les veut aussi indépendantes des variables courantes de la partie, 
    même si c'était PRATIQUE avant de définir les arbres de jeu, de les avoir dans la 
    classe Partie"""



import copy as c
import random as rd


class Regle:

    partie_gagnante=[[1,2,3], [4,5,6], [7,8,9], [1,4,7], [2,5,8], [3,6,9], [1,5,9], [3,5,7]]
    
    def etat_case(self, positions, case_en_cours, pion): 
        
        etat=[]
        for x in self.partie_gagnante:
            if all(y in positions[case_en_cours][pion] for y in x):
                # renvoie pion s'il y a une combinaison qui est satisfaite
                etat = pion

        pos_totale_c_en_cours = positions[case_en_cours]['X']+positions[case_en_cours]['O']
        
        if etat == [] and len(pos_totale_c_en_cours) == 9:
            #renvoie 'nul' dès lors que la case est remplie.
            etat = "nul"
            
        
        return etat
        
        
    def est_gagnant(self, positions: {int : [str]}, case_en_cours : int, pion : str): 
        """"-> bool"""

        for x in self.partie_gagnante:
            if all(positions[y]["gagnant"]==pion for y in x):
                # renvoie True s'il y a une combinaison qui est satisfaite
                return True

        # retour False s'il n'y a pas d'alignement
        return False

    def est_nul(self, positions: {int : [str]}, case_en_cours : int, pion : str) : -> bool

        
        if all(positions[X]["gagnant"] != [] for X in positions.keys()): 
            # il n'y a plus aucune valeur égale à une liste vide
            #renvoie True s'il n'y a donc plus de case jouable.
            return True

        return False
    
    def nb_coups(self, case_en_cours, coups):

        nb = len(coups[case_en_cours])
        return nb



class Joueur:
    pion = 'n'
    adversaire= 'w'

    def __init__(self, pion):
        self.pion = pion
        self.adversaire = "X" if self.pion=="O" else "X"
        
    def ouverture(self,positions):
        pass
        
    def tour(self,case_precedente, case_en_cours, plateau, positions, coups):
        pass
        
    def adversaire(self):
        if self.pion == "X":
            return "O"
        else:
            return "X" 
            


"""LES AGENTS :
___________________________"""

"""
1/Selection:
In this step, the algorithm traverses the tree from the root to a leaf node based 
on some selection policy (often using the Upper Confidence Bound (UCB) formula).
The selection process depends on the current state of the tree and the available actions.

2/Expansion:
Once a leaf node is reached, the algorithm expands it by adding child nodes corresponding 
to unexplored actions.The expansion step relies on the available actions and the current 
state.

3/Simulation (Rollout):
During simulation, the algorithm plays out a random or heuristic game from the newly 
expanded node to a terminal state. The quality of the simulation affects the overall
performance of MCTS.

4/Backpropagation:
After the simulation, the results (such as win/loss outcomes) are backpropagated up the tree.
The backpropagation step updates the statistics (visit count, total reward) for each node 
along the path taken during selection. The interdependency lies in how the results are 
propagated and how the tree structure is updated.
The steps are indeed interdependent:

Selection and Expansion rely on the current state and available actions.
Simulation depends on the expanded node and the chosen rollout policy.
Backpropagation updates the tree based on the simulation results.
In summary, the steps of MCTS are tightly connected, and their effectiveness 
collectively determines the algorithm’s performance. Each step informs the others,
leading to an iterative improvement in decision-making. 🌳🎲

"""
import h5py as h
import numpy as np



class Attribut :
    """
    Constante : pour les jeux à somme nulle, on a dans l'UCBT :
    - c = sqrt(2) 
    
    Fonctions :
    -etat (positions, pion) -> renvoie l'état de la position en cours.
    -accessibles (tour_precedent:chaine de 2 caractères, arbre, chemin).
    -UCBT qui prend en entrée la valeur de l'attr. UCBT d'un noeud et 
    renvoie la valeur de la UCBT associée au noeud.
    """
    
    """calculs d'évaluation et réévaluation."""
    
    c=np.sqrt(2)

    def etat (self, positions : {int: [str]}, pion : str): ->  [int]
        """etat: un tableau 1D suivant : np.array([0,-1,2,0,0,1,1,0,-1]) avec 
        |-1: victoires adversaire
        |2 : victoires joueur
        |1 : match nul
        |0 : etat non déclaré."""
        
        e = positions[i]['gagnant']
        
        etat=[(0 if e == [] else (2 if e == pion else (1 if e =='nul' else -1))) 
                for i in range(1,10)]
        
        return etat
    
    def accessibles (self, tour_precdt : str, arbre_partie, chemin : str): -> [str]
        
        #on accède au précédent attribut accessibles pour obtenir le nouveau.
        
        cases_acces = list(arbre_partie[chemin].parent.attrs.__getitem__('accessibles'))
        
        if tour_precdt != None :
            #ASCII--> numero lettre dans l'alphabet
            i,j= ord(tour_precdt[0])-65, tour_precdt[1] 
            
            cases_acces[i] = cases_acces[i].replace(j,'')
            
            return cases_acces
            
        return ['123456789' for i in range(9)]

    def UCBT (self, nb_vict : int, n : int, N : int, adv : bool ): -> float
        
        if n ==0:
            #valeurs très grandes pour forcer MCTS à exploré les noeuds nouvellement visités.
            return float('-inf') if adv else float('+inf') 
            
        return nb_vict/n +self.c*np.sqrt(np.log(N)/n)
    
   
    
    
class Arbre:
    """
    Fonctions :
    
    -cree_un_arbre(nom, etat, adv) qui initialise un arbre. Elle renvoie une racine.
    
    -ajout_fils_si_pas_existant(groupe, nom, accessibles, etat, adv). Elle ajoute au groupe 
    un nouveau fils s'il n'existe pas déjà, au nom qu'on lui donne.
    
    -retropropagation : elle prend en paramètre une feuille et elle actualise la valeur
     de l'attr. UCBT de ses prédécesseurs, ainsi que celle des voisins (de même noeud parent).
    
    - elagage (arbre, chemin, pos_choisie) enlève les branches non choisies d'un arbre mis
     en paramètre. Elle conserve uniquement la branche donnant sur position_choisie.

    
    """
    
    def cree_un_arbre(self, nom : str, etat : <class 'NoneType'>, adv : bool): 
        
        """"-> <class 'h5py._hl.files.File'>"""
        
        if etat == None :
            etat = [0 for i in range(9)]
            
        r = 'E:/UTTT/TIPE 3juin _ ordi CDI.py'

        racine = h.File(r+nom, mode = 'a')
        racine.attrs.create(name = "etat", data = etat)
        racine.attrs.create(name = "accessibles", data = ['123456789' for i in range(9)])
        racine.attrs.create(name = 'UCBT', data = (0,0,0))
        racine.attrs.create(name = 'adv', data = adv)
        
        return racine
    
    def ajout_fils_si_pas_existant (self,groupe, nom:str, acces:list, etat:list, adv:bool): 
        """"-> ()"""
        b = groupe.require_group (nom)
        
        if type(acces)==list :
            b.attrs.create(name="accessibles", data=acces)
        if type(etat) == list:
            b.attrs.create(name ="etat", data = etat)
        
        b.attrs.create('UCBT',(0,0,0))
        
        b.attrs.create('adv', adv)

    def retropropagation(self, feuille : <class 'h5py._hl.files.File'>): -> ()
        i=0
        w, n, N = feuille.attrs.__getitem__('UCBT')
        noeud = feuille
        
        while i == 0:
            noeud = noeud.parent
            if N> 0:
                for fils in noeud.keys():
                    if 'UCBT' in noeud[fils].attrs.keys():
                        v,m,M = noeud[fils].attrs.__getitem__('UCBT')
                        noeud[fils].attrs.modify('UCBT',(v,m,N+M))
            
            if 'UCBT' in noeud.attrs.keys():
                v,m,M = noeud.attrs.__getitem__('UCBT')
            else :
                v, m, M = 0,0,0 
            noeud.attrs.modify('UCBT', (w+v, m+n, M+n)) 
            
            if noeud.name == '/': #condition de sortie de boucle == on a atteint la racine.
            
                i=1

    def elagage(self, arbre, chemin : str, position_choisie : str): -> ()
        #principe : on a un arbre, un chemin dans cet arbre, et une position depuis le noueud de fin de ce chemin.*
        # On souhaite supprimer tous les chemins parcourus au delà du noeud terminant chemin, qui ne sont pas depuis "position_choisie".
        
        noeud_courant = arbre[chemin]
        
        for i in noeud_courant.keys():
            if i != position_choisie:
                noeud_courant.__delitem__(i)



class MCTS (Joueur):

    """
    Constante: 
    
    cases_gp = un dictionnaire répertoriant les valeurs char correspondant
    aux cases du grand plateau.
    
    Variables :

    arbre_partie = un arbre total. qui s'actualise en place.
    noeud_courant = le noeud sur lequel on travaille. Il est dans l'arbre_partie.
    
    chemin_ partie = le chemin entre la racine et la position en cours

    Fonctions :
    
    -ouverture ( positions) -> renvoie un numéro de sous-plateau dans lequel jouer.
    
    -tour : elle actualise l'arbre et le chemin en ajoutant le coup précédent
     fait par l'adversaire et renvoie le coup choisi par l'agent. 

    est_fini : elle renvoie un bool pour dire s'il y a ou non (un match nul ou une victoire).

    maximise_UCBT (self, L_trad, courant) : elle choisit, si il y a, le groupe suivant 
    (dont le chemin correspond à sa position) le plus prom  etteur, selon l'UCBT.
        L_trad est l'ensemble des chaines de caractère (une lettre et un chiffre) 
        correspondant à toutes les possibles positions que le joueur peut prendre 
        lors de son tour.


    """
    
    
    A=Arbre()
    arbre_partie = A.cree_un_arbre('racine'+str(rd.randint(0,100)), None, False)
    noeud_courant = arbre_partie
    cases_gp = dict(zip(range(1,10),"ABCDEFGHI"))

    chemin_partie = ""
    
    
    def ouverture(self, positions : {int : [str]}): -> (int, int)
        
        L = [i for i in range (1,10) if positions[i]["gagnant"] == []]
        return rd.choice(L)   #random
    
    def tour (self, case_pr, case_en_cours, plateau:dict, positions:dict, coups : dict): 
        """"-> (int,int)"""
        A= Arbre()
        Att = Attribut ()

        if case_pr == None :
            #cas où la partie vient de commencer.
            adv = False
        else :
            #cas où le joueur adverse a joué avant.
            adv = True
            
            tour_precdt = self.cases_gp [case_pr] + str(case_en_cours)
            self.chemin_partie += "/"+ tour_precdt
            
            etat = Att.etat(positions, self.pion)
            
            A.ajout_fils_si_pas_existant(self.noeud_courant, tour_precdt, None, etat, adv)
            
            Acc = Att.accessibles(tour_precdt, self.arbre_partie, self.chemin_partie)
            
            #Actualiser l'arbre de jeu.
            self.arbre_partie[self.chemin_partie].attrs.create('accessibles', Acc)
            self.noeud_courant = self.arbre_partie[self.chemin_partie]
        
        #Le joueur décide d'une position à jouer :
        ci,cj = self.decision (case_en_cours, positions, coups) 
        
        #On actualise l'arbre avec la nouvelle position jouée.
        pos1 = self.actualise_positions(positions, ci, cj, False)
        

        tour_precedent = self.cases_gp[ci] + str(cj)
        
        A.elagage(self.arbre_partie, self.chemin_ partie, tour_precedent)
        
        self.chemin_partie += '/' + tour_precedent
        
        etat = Att.etat(pos1, self.pion)
        
        A.ajout_fils_si_pas_existant(self.noeud_courant, tour_precedent, None, etat, False) 
        
        Acc = Att.accessibles(tour_precedent, self.arbre_partie, self.chemin_partie)
        
        self.arbre_partie[self.chemin_partie].attrs.create('accessibles', Acc)
        self.noeud_courant = self.arbre_partie[self.chemin_partie]
        return ci,cj
        
        
    def decision (self, case_en_cours : int, positions : dict(), coups : dict()): 
        """-> (int,int)"""
        """prend la case en cours, le dict positions et les coups en paramètres et 
        renvoie une position i,j."""
        A = Arbre()
        Att = Attribut()
        X = 8 
        for i in range(X):
            
            adv_selec = self.prochaine_valeur_adv(self.chemin_partie, positions) 
            position_choisie, chemin, fin = self.selection(adv_selec) 
            
            if len(chemin)>=3:
                N= len(self.chemin_partie)
                case, i = self.traduit_nom_position_en_coordonnee(chemin[:N+3])
                if type (fin) == int:
                    
                    if fin == 1:
                        # on renvoie la valeur car c'est la meilleure possible.
                        return case, i 
                
            pos1, cou1 =self.actualise_positions_coups(chemin, positions, coups, adv_selec)
            adv_expl = not(self.prochaine_valeur_adv(chemin, pos1))
                
            if self.exploration(chemin, pos1):
                
                print("on a exploré!")
                
                courant = self.arbre_partie[chemin] if chemin != '' else self.arbre_partie
                p = len(courant.keys())
                
                for j in courant.keys():
                    case_, i_ = (ord(j[0])-64, int(j[1]))
                    
                    W,S = self.simulation (case_, i_, pos1, cou1, adv_expl) 
                    # S est une constante de simulations
                    C = courant [j]
                    C.attrs.modify('UCBT', (W, S, p*S))

                    A.retropropagation(C)
                    
        pos_choisie2, chemin2, fin2 = self.selection(adv_selec)
        
        if len(chemin2)>= N+3 :
            c2, i2 = self.traduit_nom_position_en_coordonnee(chemin2[:N+3])
            return c2, i2
            
        else :
            return case, i
            
    def selection (self, adv : bool): -> (str, str, int) 
        """Cette fonction sert à parcourir les noeuds les plus prometteurs
        (ou bien ceux qu'il faudrait continuer d'approfondir car il leur reste
         du potentiel à découvrir) déjà explorés.
        Elle renvoit le chemin de la feuille de plus haut potentiel 
        selon l'heuristique UCBT"""
        
        # renvoie le chemin parcouru et un paramètre fin qui est soit None 
        # (si ce n'est pas un etat final), soit egal à la valeur 0,1,-1 en
        # fonction de l'état de fin.
        R = Regle()
        chemin = self.chemin_partie
        
        nom_pos = self.choisit(chemin, adv)
        
        if chemin != '':
            courant = self.arbre_partie[chemin]
        else :
            courant = self.arbre_partie
        etat_jeu = self.etat_total_plateau_position(nom_pos,chemin)
        
        while nom_pos != None and etat_jeu == None :
            chemin += "/"+nom_pos
            adv = not(adv)
            courant = self.arbre_partie[chemin[1:]]
            nom_pos = self.choisit(chemin, adv)
            etat_jeu = self.etat_total_plateau_position(nom_pos,chemin)
        
        if nom_pos == None :
            return nom_pos, chemin, etat_jeu
        else:
            chemin += '/' + nom_pos
            
            return nom_pos, chemin, etat_jeu
            
    
    
    def choisit (self, chemin : str, adv : bool ): -> str 
        
        if chemin !='':
            courant = self.arbre_partie[chemin]
        else :
            courant =self.arbre_partie
        
        if courant.attrs.__contains__('accessibles'):
            
            Acc = self.positions_accessibles (chemin)
            L_trad = self.traduit_liste_couples_en_caracteres (Acc)
            
            i = self.maximise_UCBT(L_trad, courant, adv)
            print(i, " position qui maximise l_UCBT")
            return i
        else :
            k = list(courant.keys())
            if k == []:
                return None
            else :
                return 'Erreur : existence d_un noeud exploré sans UCBT '
        
    
    def exploration (self, chemin : str, positions : dict()): -> bool
        A = Arbre()
        Att = Attribut()
        accessibles = self.positions_accessibles(chemin)
        if accessibles == []:
            # si jamais il ne fait rien, alors il renvoie un booléen de sorte qu'on peut
            #savoir qu'il n'y a pas eu de modif'
            return False 
        n = len(accessibles)
        ind = [i for i in range(n)]
        p = n // 3 if n>= 8 else n  # arbitraire : je ne veux pas explorer trop de noeuds
                                    # pour limiter la complexité spatiale et temporelle
        L = [rd.choice(ind) for i in range(p)]
        courant = self.arbre_partie[chemin] if chemin != '' else self.arbre_partie
        adv = self.prochaine_valeur_adv(chemin, positions)
        
        for j in L:
            c, i = accessibles[j]
            nom = self.cases_gp[c] +str(i)
            pos1 = self.actualise_positions(positions, c, i, adv)
            A.ajout_fils_si_pas_existant(courant, nom, None, Att.etat(pos1, self.pion), adv)
            Acc = Att.accessibles(nom, self.arbre_partie, chemin +'/'+nom)
            self.arbre_partie[chemin +'/'+ nom].attrs.create('accessibles', Acc )
        return True
    
    def simulation(s, case : int, i : int, pos : dict, cou : dict, adv : bool): 
        """-> (int,int)"""
        # simule une partie et renvoie une évaluation.// 
        #/!\ : les dictionnaires doivent être des deepcopy
        """principe : on doit opérer plusieurs parties aléatoires depuis la position de 
        laquelle on est dans la partie. /!\ On doit veiller à ce que celle-ci soit bien 
        encore stockée (tour) et qu'on ne l'ait pas modifiée par inadvertance.
        On doit aussi implémenter de manière très efficace afin que les parties simulées 
        puissent ne pas prendre trop d'espace temporel et spatial."""
        
        pion = s.pion if not(adv) else s.adversaire
        
        S = 25
        R = Regle()
        pos1 = c.deepcopy(pos)
        cou1 = c.deepcopy(cou)

        pos1[case][pion].append(i+1) # /!\ - Ici on écrit de 1,..,9
        cou1[case].append(i+1)

        et = R.etat_case(pos1, case, pion)
        
        pos1[case]["gagnant"] = et

        if et != [] and R.est_gagnant(pos1, case, pion): 
        
            return (0 if adv else S, S )
            
        else :
            W=0
            for j in range (S): 
                pos2 = c.deepcopy(pos1)
                W+=1 if s.simu(pos2, i-1, case, c.deepcopy(cou1), adv) == 1 else 0 
                # on écrit i-1 car on se met aux normes de prochain_tour
                
            return W, S

    
    def simu(self, pos : dict, i : int, case : int, cou : dict, adv : bool) : -> int
        # simule une partie et renvoie une évaluation.
        R=Regle()
        pion= self.pion if not(adv) else self.adversaire    
        
        while not(self.est_fini(pos, case, pion)) :
            pion=self.adversaire if pion==self.pion else self.pion
            prochain_ = self.prochain_tour(i+1, pos, cou) #coups_accessible
            
            if prochain_[0][1]==[]:
                print('err MCTS prochain_tour', pos, cou, i+1)
                raise ValueError
            else :
                case,list_ind = rd.choice(prochain_)
                i = rd.choice(list_ind)
                pos[case][pion]. append(i+1)
                cou[case].append(i+1)
                pos[case]["gagnant"]=R.etat_case(pos, case, pion)

        if R.est_gagnant(pos, case, self.pion):
            return +1
        return -1
    
    def est_fini(self, position : dict(), case_en_cours : int, pion : str): -> bool
        R=Regle()
        c = case_en_cours
        if R.est_gagnant(position, c, pion) or R.est_nul(position, c, pion):
            return True
         return False


    def maximise_UCBT (self, L_trad : [str], courant, adv : bool): -> str
        """L_trad est une liste de nom_positions telles que ces positions sont susceptibles
         d'être filles du noeud_courant, et sont des positions licites. 
         Courant est le noueud_courant de la partie."""
        
        """Rq : si on est sur un etage de notre arbre qui est un tour de l'adverse, 
        alors, si l'adv joue pour gagner, ops qu'il joue de telle sorte à minimiser nos
        chances de gagner nous.
         Cependant, ce n'est pas vraiment symétrique. En effet, alors que nous jouons 
         pour gagner, on suppose que l'adverse joue au moins pour nous empêcher de gagner, 
         en prenant le min de l'UCBT. Car dans le min : défaite + match nuls"""
        
        Att = Attribut()
        max_UCBT = float('-inf') if not(adv) else float('+inf')
        max_pos = None
        L_contenue = [i for i in L_trad if courant.__contains__(i)]
        for i in L_contenue:
                c = courant[i]
                if c.attrs.__contains__('UCBT') :
                    n_vict, n, N = c.attrs.__getitem__('UCBT')
                    UCBT= Att.UCBT(n_vict, n, N, adv)
                    if (UCBT >=max_UCBT and not(adv)) or  (UCBT <=max_UCBT and adv):
                        print(max_UCBT, max_pos, adv)
                        max_UCBT = UCBT
                        max_pos = i
                
                else : # il faut initialiser l'attribut 
                    c.attrs.create('UCBT',(0,0,0))
                    n_vict, n, N = c.attrs.__getitem__('UCBT')
                    UCBT= Att.UCBT(n_vict, n, N, adv)
                    if (UCBT >=max_UCBT and not(adv)) or  (UCBT <=max_UCBT and adv) :
                        max_UCBT = UCBT
                        max_pos = i
        return max_pos
    

    def etat_total_plateau_position (self, nom_pos : str, chemin : str): -> int
        
        """Le principe de cette fonction est de calculer l'état total du plateau dans
         une position donnée. Il nous permet de savoir quand la partie se termine ou non."""
        
        if chemin != '':
            courant = self.arbre_partie[chemin if nom_pos==None else chemin+'/'+nom_pos ]
        else :
            courant = self.arbre_partie
        
        R = Regle()
        
        
        for combinaison in R.partie_gagnante :
            etat= courant.attrs.__getitem__('etat')
            
            if all (etat[x-1] == 2 for x in combinaison) :
                return  1 # on gagne
                
                
            elif all(etat[x-1]==-1 for x in combinaison) :
                return  -1 # adv gagnant
                
            elif all(x != 0 for x in etat):
                return 0 # match nul
            else :
                return None
    
    
    def traduit_nom_position_en_coordonnee(self, chemin : str): -> (int, int)
        
        case, i = ord(chemin[-2])-64, int(chemin[-1])
        return case, i
    
    
    def prochaine_valeur_adv(self, chemin str, positions : dict()): -> bool
        
        if chemin== "":
            courant = self.arbre_partie
        else :
            courant = self.arbre_partie[chemin[1:]]
        
        if courant.attrs.__contains__('adv'):
            adv = not(courant.attrs.__getitem__('adv'))
        
        else :
            print("pb adv : l_adv du parent non def")
            c,i = self.traduit_nom_position_en_coordonnee(chemin[:3])
            
            
            if i in positions[c][self.pion]:
                #le 1er joueur = nous
                if (len(chemin)//3)%2 == 0:
                    
                    adv = False # la précédente position a été jouée par l'adv
                else :
                    adv = True # la précédente position a été jouée par MCTS
            else:
                #le 1er joueur = l'autre
                if (len(chemin)//3)%2 == 1:
                    adv = False # la précédente position a été jouée par l'adv
                else :
                    adv = True # la précédente position a été jouée par MCTS
        
        return adv
        
    
    def positions_accessibles(self, chemin : str): -> [(int,int)]
        
        if chemin != '':
            groupe = self.arbre_partie [chemin]
        else :
            groupe = self.arbre_partie
            
        etat = groupe.attrs.__getitem__('etat')

        if len(chemin)>1:
            c = int(chemin[-1])

            if etat[c-1] == 0: #2, 1 et -1 sont pour les états déclarés
                acces = groupe.attrs.__getitem__('accessibles')
                acc= acces[c-1]
                accessibles = [(c,int(i)) for i in acc]
                return accessibles

            n = len(etat)
            accessibles = []
            for j in range(n):
                if etat[j] == 0:
                    acces = groupe.attrs.__getitem__('accessibles')
                    acc = acces[j]
                    accessibles += [(j+1,int(i)) for i in acc] 
                    #liste contenant des couples c,i (case_gp, case_sp)

            return accessibles #liste contenant des couples c,i (case_gp, case_sp)

        return [(i, j) for i in range (1,10) for j in range(1,10)]
    
    
    
    def traduit_liste_couples_en_caracteres(self, pos_accessibles : [(int,int)]): -> [str]
        
        p = 1
        liste_pos_str = []
        
        sous_plateau = self.cases_gp[p]
        for c,j in pos_accessibles:
            if p == c :
                liste_pos_str. append(sous_plateau+str(j))
            else :
                p = c
                sous_plateau = self.cases_gp[p]
                liste_pos_str.append(sous_plateau +str(j))

        return liste_pos_str
    
    
    def actualise_positions_coups (s, chemin_sel, positions:dict, coups:dict, adv_sel:bool):
        """" -> dict(), dict()"""
        
        R =Regle()
        
        N = len(s.chemin_partie)
        chemin_suppl = chemin_selec[N:] if N>0 else chemin_sel
        
        pos = c.deepcopy(positions)
        cou = c.deepcopy(coups)
        
        
        L = str.split(chemin_suppl, '/')
        pion = s.adversaire if adv_sel else s.pion
        
        for j in L :
            
            if j != "":
                print(j)
                ca, i = s.traduit_nom_position_en_coordonnee(j)
                pos[ca][pion].append(i)
                pos[ca]["gagnant"] = R.etat_case(pos, ca, pion)
                pion = s.adversaire if pion == s.pion else self.pion
            
        return pos, cou
        

    def actualise_positions(self, positions : dict, c_ : int, i : int, adv : bool):
        """ -> dict()"""
        R= Regle()
        
        pion = self.adversaire if adv else self.pion
        pos = c.deepcopy(positions)
       
        pos[c_][pion].append(i)
        pos[c_]["gagnant"] = R.etat_case(pos, c_, pion)
        return pos
        
    def actualise_coups( self, coups : dict, c_ : int, i :int): -> dict
        cou = c.deepcopy(coups)
        cou [c_].append(i)
        return cou
    
    def simu(self, pos : dict, i : int, case : int, cou : dict, adv : bool) : -> int
        # simule une partie et renvoie une évaluation.
        R=Regle()
        
        pion= self.pion if not(adv) else self.adversaire    
        
        while not(self.est_fini(pos, case, pion)) :
            pion=self.adversaire if pion==self.pion else self.pion
            prochain_ = self.prochain_tour(i+1, pos, cou) #coups_accessible
            
            if prochain_[0][1]==[]:
                print('err MCTS prochain_tour', pos, cou, i+1)
                raise ValueError
            else :
                case,list_ind = rd.choice(prochain_)
                i = rd.choice(list_ind)
                pos[case][pion]. append(i+1)
                cou[case].append(i+1)
                
                pos[case]["gagnant"]=R.etat_case(pos, case, pion)
        if R.est_gagnant(pos, case, self.pion):
            return +1
        return -1

    


    def prochain_tour(self, case_en_cours : int, positions : dict, coups : dict: 
        """-> (c_gp * (c_sp list)) list"""
        cases_libres=[(None,[])]
        c = case_en_cours 

        if positions[c] == []:

            cases_libres = [(c, [j-1 for j in range(1,10) if j not in coups[c]])]

        if cases_libres[0][1] == []:
            sp = coups.keys()
            etat_sp = [positions[i]["gagnant"] for i in sp]
            
            cases_libres = [(i,[k-1 for k in range(1,10) if k not in coups[i]]) for i in 
                            sp if etat_sp[i] == [] and len(coups[i])<9)]
            return cases_libres
        if cases_libres[0][0] != None:
            
            return cases_libres
        else :
            raise ValueError



#_______________________________________________________________________________________________________________

class Montecarlo(Joueur):
    
    def ouverture(self,positions):
        
        L = [i for i in range (1,10) if positions[i]["gagnant"] == []]
        return rd.choice(L)

    def prochain_tour(self, case_en_cours : int, positions : dict(), coups : dict()): 
        """"-> (c_gp * (c_sp list)) list"""
        cases_libres=[(None,[])]
        c = case_en_cours

        if case_en_cours != None:
            cases_libres = [(c, [j-1 for j in range(1,10) if j not in coups[c]])]

        if c == None or cases_libres[0][1] == []:
            cases_libres=[(i,[k-1 for k in range(1,10) if k not in coups[i]]) for i in 
                    coups.keys() if (positions[i]["gagnant"] == [] and len(coups[i])<9)]
        if cases_libres[0][0] != None:
            return cases_libres
        else :
            raise ValueError
            
    def est_fini(self, position : dict(), case_en_cours : int, pion : str): -> bool
        R=Regle()
        c = case_en_cours
        if R.est_gagnant(position, c, pion) or R.est_nul(position, c, self.pion):
            return True
        return False

    def simulation(self, case : int, i : int, pos : dict(), cou : dict()): -> int
        """case de 1,...,9 et i de 0,...,8"""
        R = Regle()
        pos1 = c.deepcopy(pos)
        cou1 = c.deepcopy(cou)

        pos1[case][self.pion].append(i+1) # /!\ - Ici on écrit de 1,..,9
        cou1[case].append(i+1)

        et = R.etat_case(pos1, case, self.pion)
        pos1[case]["gagnant"] = et

        if et != [] and R.est_gagnant(pos1, case, self.pion): 
            return float("+inf")

        else :
            s=0
            for j in range (200):
                s += self.simu( c.deepcopy(pos1), i, case, c.deepcopy(cou1))
            return s/200


    def simu(self, pos, i, case, cou) : 
        # simule une partie et renvoie une évaluation.        
        R=Regle()
        pion= self.pion
        while not(self.est_fini(pos, case, pion)) :

            pion=self.adversaire if pion==self.pion else self.pion

            prochain_ = self.prochain_tour(i+1, pos, cou) #coups_accessible
            if prochain_[0][1] == []:
                
                print("erreur MC_prochain", pos, cou)
                raise ValueError
            else :
                case, list_ind = rd.choice(prochain_)
                i = rd.choice(list_ind)
    
                pos[case][pion]. append(i+1)
                cou[case].append(i+1)
                
                pos[case]["gagnant"]=R.etat_case(pos, case, pion)
                
                if pos[case]["gagnant"] != []:
                    case = None
        if R.est_gagnant(pos, case, self.pion):
            return +10
        if R.est_gagnant(pos, case, self.adversaire):
            return -10
        else:
            return +1

    def tour(self, case_pr, case_en_cours, plateau, positions, coups):
        
        pos1 = c.deepcopy(positions)
        cou = c.deepcopy(coups) 

        cases_libres = self.prochain_tour (case_en_cours, pos1, cou)
        
        s = self.simulation
        
        nb_victoire = [ [(s(ca, i, c.copy(pos1), c.copy(cou)), ca, i) for i in sc_libres] 
                        for (ca,sc_libres) in cases_libres if sc_libres != [] ]
                         
        # list of list de 3-uplets (valeur simu, c_gp, i

        c_max,i_max = None, None
        victoire_max = float("-inf")
        for list_c_gp in nb_victoire:
            m, ca, i = max (list_c_gp)
            if m > victoire_max:
                victoire_max, c_max, i_max = m, ca, i

        return c_max, i_max+1



class Aleatoire(Joueur):
    #computer player whose strategy is to play random box

    def ouverture(self,positions):
        
        L = [i for i in range (1,10) if positions[i]["gagnant"] == []]
        return rd.choice(L)
        
    def tour(self,case_pr, case_en_cours, plateau, positions, coups):
        
        if positions[case_en_cours]==[]:
            
            cases_prises = coups[case_en_cours]
            cases_libres = [i for i in range(1,10) if i not in cases_prises]
            return case_en_cours, rd.choice(cases_libres)
        
        else :
            cases_prises = coups[case_en_cours]
            cases_libres =[(c_, [i for i in range(1,10) if i not in coups[c_]])
                            for c_ in range(1,10) if positions[c_]["gagnant"]==[]]
            
            case, L_ind = rd.choice(cases_libres)
            
            return case, rd.choice(L_ind)
        
        



"""
_________________________
CLASSE PARTIE :
"""


class Morpion:

    # Représente le plateau du Super Morpion
    
    #dictionnaire de clés 1,..,9 avec
    #pour valeurs des listes représentant les sous_plateaux.
    #On stocke des caractères ' '/'X'/'O'
    plateau = { i : j for i,j in zip( [i+1 for i in range(9)], 
                    [[' ' for i in range(9)] for x in range(9)]) }
    
    coups = {i: j for i,j in zip( [i for i in range(1,10)], [[] for x in range(9)])}
    
    #On stocke dans chaque clé (de 1 à 9) les positions acquises par "X", "O",
    #et l'état du sous-plateau (ie "X"/"O" a gagné, match "nul" ou encore [] si pas encore déclaré.)
    #La valeur d'une clé est de type dict afin de rendre la structure plus simple à manier.
    positions = {i: {"X": [], "O": [], "gagnant":[]} for i in range(1, 10)}
    

    case_en_cours = None
    case_precedente = None
    en_cours = "joueur1"
    joueurs = [Joueur, Joueur]


    def init_plateau(self):

        #On initialise le plateau, ainsi que la variable case_en_cours 
        #(case du plateau globale courante), les variables paramétrant le plateau.
        
        self.case_en_cours = None
        self.plateau = { i : j for i,j in zip( [i for i in range(1,10)],
                        [[' ' for i in range(9)] for x in range(9)]) }
        self.affichage_plateau()
        self.positions = {i: {"X" : [], "O" : [], "gagnant" : []} for i in range(1, 10)} 
        self.coups = {i : j for i,j in
                        zip( [i for i in range(1,10)], [[] for x in range(9)])}

        #On identifie les joueurs s'opposant:
        if self.en_cours == "alea":
            self.joueurs[0] = Aleatoire("X")
            self.joueurs[1] = Aleatoire("O")
        if self.en_cours == "MC-alea":
            self.joueurs[0] = Montecarlo("X")
            self.joueurs[1] = Aleatoire("O")
        if self.en_cours == "alea-MC":
            self.joueurs[0] = Aleatoire("X")
            self.joueurs[1] = Montecarlo("O")
        if self.en_cours == "alea_mcts":
            self.joueurs[0] = Aleatoire("X")
            self.joueurs[1] = MCTS("O")
        if self.en_cours == "mcts_alea":
            self.joueurs[0] = MCTS("X")
            self.joueurs[1] = Aleatoire("O")
        if self.en_cours == "ordi":
            self.joueurs[0] = Montecarlo("X")
            self.joueurs[1] = Montecarlo("O")
        elif self.en_cours == 'joueur':
            self.joueurs[0] = Montecarlo("X")
            self.joueurs[1] = MCTS("O")
        elif self.en_cours == 'joueur1':
            self.joueurs[0] = MCTS("X")
            self.joueurs[1] = Montecarlo("O")
        elif self.en_cours == 'MCTS':
            self.joueurs[0] = MCTS("X")
            self.joueurs[1] = MCTS("O")


    # Affiche le plateau du Super Morpion
    def affichage_plateau(self):
        p=self.plateau_() #liste des positions des sous-plateaux
        for m in range (3):
            for k in range (3):
                for i in range(3*m,3*m+3):

                    for j in range(3*k,3*k +3):
                        print(p[i][j],end=" ")
                    if i-3*m <2:
                        print("|",end=" ")
                print()

            print("------|-------|------")

    #crée une liste des sous-plateaux
    def plateau_(self):
        plat=[]
        p=self.plateau
        for i in p.keys():
            a=p[i]
            plat.append(a)
        return plat

    #affiche classiquement le sous-plateau en cours:
    def affichage_sous_plateau(self):
        if self.case_en_cours != None:
            print("\n")
            print("\t     |     |")
            print("\t  {}  |  {}  |  {}".format(self.plateau[self.case_en_cours][0],
             self.plateau[self.case_en_cours][1], self.plateau[self.case_en_cours][2]))
            print('\t_____|_____|_____')

            print("\t     |     |")
            print("\t  {}  |  {}  |  {}".format(self.plateau[self.case_en_cours][3], 
            self.plateau[self.case_en_cours][4], self.plateau[self.case_en_cours][5]))
            print('\t_____|_____|_____')

            print("\t     |     |")

            print("\t  {}  |  {}  |  {}".format(self.plateau[self.case_en_cours][6], 
            self.plateau[self.case_en_cours][7], self.plateau[self.case_en_cours][8]))
            print("\t     |     |")
            print("\n")
    
    

    #Lance une partie de Super Morpion.
    def partie(self):

        R=Regle()
        self.init_plateau()
        chemin=''
        print("Nouvelle partie, commençant par "+self.en_cours+" avec "+self.joueurs[0].pion)
        
        # Boucle de jeu d'une partie unique de Super Morpion
        while True:
            
            if self.case_en_cours == None:
                self.case_en_cours=self.joueurs[0].ouverture(self.positions)
            
            c_p, c_c = self.case_precedente, self.case_en_cours
            plt = self.plateau
            pos = self.positions
            coups = self.coups

            self.case_en_cours, move = self.joueurs[0].tour(c_p, c_c, plt, pos, coups)
            
            print("la position",(self.case_en_cours, move),"a été choisie par", self.joueurs[0].pion)

            # Contrôle de la validité d'une valeur entrée.
            if move < 1 or move > 9:
                
                raise Exception("Wrong Input!!! Try Again")
            
            # Contrôle si la position entrée est effectivement libre
            if self.plateau[self.case_en_cours][move - 1] != " ":
                
                raise Exception("Place already filled. Try again!!")

            # Màj à jour des informations du jeu
            # Màj de l'état du plateau:
            chemin += "/"+str(self.case_en_cours)+str(move)
            self.coups[self.case_en_cours].append(move)
            self.plateau[self.case_en_cours][move - 1] = self.joueurs[0].pion
            print(self.joueurs[0].pion,"joue",move)
            # Màj des positions des joueurs:
            self.positions[self.case_en_cours][self.joueurs[0].pion].append(move)

            etat= R.etat_case( self.positions, self.case_en_cours, self.joueurs[0].pion)

            if etat != []:
                self.positions[self.case_en_cours]["gagnant"] = etat
                Est_Gagnant = R.est_gagnant(self.positions, self.case_en_cours, self.joueurs[0].pion)
                Est_Nul = R.est_nul(self.positions, self.case_en_cours, self.joueurs[0].pion)

                # Contrôle l'état de la partie -> tests de fin de partie
                if Est_Gagnant:
                    dictionnaire_victoire[self.joueurs[0].pion].append(chemin)
                    self.affichage_plateau()
                    return [self.joueurs[0].pion, self.coups, self.plateau]
                if Est_Nul:
                    dictionnaire_victoire['nul'].append(chemin)
                    self.affichage_plateau()
                    return ['D', self.coups, self.plateau]

            # Inversion du rôle des joueurs -> tour suivant.
            
            self.case_precedente = self.case_en_cours
            self.case_en_cours = move
            self.change_les_joueurs()
            
            print(self.joueurs[0].pion, "va pouvoir jouer à son tour")
            


    
    
    def change_les_joueurs(self):
        self.joueurs[0], self.joueurs[1] = self.joueurs[1], self.joueurs[0]
        


    def nb_coups(self):
        return len(self.positions[self.case_en_cours]['X']) + 
                len(self.positions[self.case_en_cours]['O'])
    
    def est_libre(self, position):
        return self.plateau[self.case_en_cours][position - 1] == " "



dictionnaire_victoire ={'X':[],'O':[], 'nul': []}


for i in range (1):
    mo= Morpion()
    mo.partie()


