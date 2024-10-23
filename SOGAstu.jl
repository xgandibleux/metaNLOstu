# Cours      : Metaheuristiques
# Algorithme : Resolution par AG de la fonction de Shubert
# Auteur     : Xavier Gandibleux
# Date       : 2017 - rev aout 2020 pour V1.5

@static if VERSION < v"1.5-"
           error("NOT COMPLIANT WITH JULIA < v1.5.0")
        end

using Printf
using Statistics

# Parametres de l'algorithme genetique; ces parametres DOIVENT etre calibres!
nPop         = 10   # nombre d'individus (attention : nombre obligatoirement pair)
nGenerations = 2   # nombre de generations
probCros     = 0.5   # probabilite de crossover
probMut      = 0.5   # probabilite de mutation

# Parametres du codage binaire des variables reelles
lVar         =  15   # nbre de bits codant une variable (=> chromosome de 30 bits)
xinf         = -10   # valeur inferieure des variables
delta        =  20   # range des valeurs des variables
epsilon      =   3   # precision (digits) apres la virgule

# Variables d'environnement
struct t_environment
  # for display's options: false => mute, true => verbose
  displayPrint   :: Bool # display all the prints to follow the algorithm's activity
  displayGraphic :: Bool # display the results on a graphic
end

# Experimentation
nRun         =   1   # nombre de repetition de la resolution du probleme

# Fonction de Shubert
function shubert(x1::Float64,x2::Float64)
    sum1=0.0
    sum2=0.0
    for i=1:5
        sum1 = sum1 + i * cos((i+1)*x1+i)
        sum2 = sum2 + i * cos((i+1)*x2+i)
    end
    return sum1 * sum2
end

# Decoder des variables x1, x2 d'un chromosome
function decoderVariables(x)
    # codage binaire avec bit de poids faible a gauche du mot
    x1 = 0.0
    x2 = 0.0
    for i=1:lVar
        x1=x1+x[i]*2^(i-1)
        x2=x2+x[lVar+i]*2^(i-1)
    end
    x1 = xinf + x1 * delta / (2^lVar - 1)
    x2 = xinf + x2 * delta / (2^lVar - 1)
    return x1, x2
end

# Evaluation de la fonction sur les 2 variables
function evaluerFonction(individu)
    x1,x2 = decoderVariables(individu)
    return shubert(x1,x2)
end

# Generation de la population initiale
function genererPopulationInitiale(env::t_environment, nPop, lVar)
    # les 2 variables sont encodees consecutivement dans un chromosome
    pop = Matrix{Int8}(undef,nPop,2*lVar)
    for i =1:nPop
        pop[i,:]= rand(0:1,2*lVar)
    end

    if env.displayPrint == true
        println("Population initiale ")
        for i=1:nPop
            x1,x2 = decoderVariables(pop[i,:])
            fx1x2 = evaluerFonction(pop[i,:])
            @printf("[%3d]  x1 = % 3.3f  x2 = % 2.3f  |  f(x1,x2) = %8.3f \n",i,x1,x2,fx1x2)
        end
    end

    return pop
end

# Affichage des informations (indice; variables x1, x2; valeur de f(x1,x2) ) d'un individu
function afficherIndividu(env::t_environment, num, individu)
    if env.displayPrint == true
        x1,x2 = decoderVariables(individu)
        findividu = evaluerFonction(individu)
        @printf("Individu  %1d       : x1 = % 3.3f  x2 = % 2.3f  |  f(x1,x2) = %8.3f \n",num,x1,x2,findividu)
    end
end

# Application d'un crossover a deux points
function crossover(env::t_environment, xParent1, xParent2)
    if rand() <= probCros
        # application d'un point de coupe aleatoire sur chaque variable
        coupe1=rand(1:lVar)
        coupe2=rand(lVar+1:2*lVar)

        a1=xParent1[1:coupe1];       b1=xParent1[coupe1+1:lVar]
        a2=xParent2[1:coupe1];       b2=xParent2[coupe1+1:lVar]

        c1=xParent1[lVar+1:coupe2];  d1=xParent1[coupe2+1:2*lVar]
        c2=xParent2[lVar+1:coupe2];  d2=xParent2[coupe2+1:2*lVar]

        xEnfant1= vcat(a1,b2,c1,d2)
        xEnfant2= vcat(a2,b1,c2,d1)

        if env.displayPrint == true
            println("crossover")
            println("  points de coupe   : ", coupe1, " ", coupe2)
            println("  Parents 1 (x1,x2) : ", xParent1[1:lVar], " ", xParent1[lVar+1:2*lVar])
            println("  Parents 2 (x1,x2) : ", xParent2[1:lVar], " ", xParent2[lVar+1:2*lVar])
            println("  Enfants 1 (x1,x2) : ", xEnfant1[1:lVar], " ", xEnfant1[lVar+1:2*lVar])
            println("  Enfants 2 (x1,x2) : ", xEnfant2[1:lVar], " ", xEnfant2[lVar+1:2*lVar])
        end

        return xEnfant1, xEnfant2
    else
        return xParent1, xParent2
    end
end

# Application d'une mutation
function mutation(env::t_environment, individu)
    if rand() <= probMut
        # application d'un flip aleatoire d'un bit sur chaque variable
        ibit1 = rand(1:lVar) ;                     ibit2 = rand(lVar+1:2*lVar)
        individu[ibit1] = (individu[ibit1]+1)%2 ;  individu[ibit2] = (individu[ibit2]+1)%2
        if env.displayPrint == true
            println("Mutation")
            println("  bits mutes  : ", ibit1, " ", ibit2)
            println("  Individu (x1,x2)  : ", individu[1:lVar], " ", individu[lVar+1:2*lVar])
        end
    end
    return individu
end


# Selection d'un parent par tournoi binaire
function selectionTournoiPopulation(env::t_environment, pop)
    i1 = rand(1:size(pop,1)) ;                 i2 = rand(1:size(pop,1))
    xParent1 = pop[i1,:] ;                     xParent2 = pop[i2,:]
    fparent1 = evaluerFonction(xParent1) ;     fparent2 = evaluerFonction(xParent2)

    if env.displayPrint == true
        @printf("Selection : [%3d] %8.3f  <->  [%3d] %8.3f \n", i1, fparent1, i2, fparent2)
    end

    if fparent1 < fparent2
        return i1, xParent1, fparent1
    else
        return i2, xParent2, fparent2
    end
end

# Selection d'un individu par tournoi binaire
function selectionTournoiSurvivant(env::t_environment, individu1, individu2)
    findividu1 = evaluerFonction(individu1) ;   findividu2 = evaluerFonction(individu2)
    if env.displayPrint == true
        @printf("Selection : %8.3f  <->  %8.3f \n", findividu1, findividu2)
    end

    if findividu1 < findividu2
        return individu1
    else
        return individu2
    end
end

# Calcul du fitness moyen de l'ensemble de la population
function fitnessAvgPopulation(pop)
    fitnessAvg = 0.0
    for i=1:size(pop,1)
      fitnessAvg = fitnessAvg + evaluerFonction(pop[i,:])
    end
    return fitnessAvg / size(pop,1)
end

# Calcul du fitness min de l'ensemble de la population
function fitnessMinPopulation(pop)
    fitnessMin = 0.0
    for i=1:size(pop,1)
      fitnessMin = min(fitnessMin, evaluerFonction(pop[i,:]))
    end
    return fitnessMin
end

function plotAvg(allMinPop, allAvgPop)
    figure("bilan valeurs moyennes toutes generations",figsize=(6,6)) # Create a new figure
    title("GA-Shubert | f(x1,x2) moyen")
    xlabel("generations")
    ylabel("f(x1,x2)")

    nPoint = length(allAvgPop)
    x=collect(0:div(nPoint,10):nPoint)
    xticks(x)
    plot(allAvgPop,linestyle="--", lw=0.5, marker="o", ms=2, color="blue", label="f(x1,x2) avg")
    plot(allMinPop,linestyle="--", lw=0.5, marker="_", ms=4, color="red", label="f(x1,x2) min")
    legend(loc=1, fontsize ="small")
end

function plotfct()
    figure("Fonction à minimiser",figsize=(6,6)) # Create a new figure
    zlabel("shubert(x1,x2)")

    n=1000
    x=range(-10, stop=10, length=n) #linspace(-10,10,n)
    xgrid = repeat(x', outer=(n,1)) # repmat(x',n,1)
    ygrid = repeat(x, outer=(1,n))  # repmat(x,1,n)
    z = zeros(n,n)
    for i=1:n
        for j=1:n
            z[i,j]=shubert(xgrid[i,j],ygrid[i,j])
        end
    end
    plot_surface(xgrid, ygrid, z, cmap="nipy_spectral", linewidth=1)
end

function initplotpop(pop)
    fig = figure()
    n=1000
    x=range(-10, stop=10, length=n) #linspace(-10,10,n)
    xgrid = repeat(x', outer=(n,1)) # repmat(x',n,1)
    ygrid = repeat(x, outer=(1,n))  # repmat(x,1,n)
    z = zeros(n,n)
    for i=1:n
        for j=1:n
            z[i,j]=shubert(xgrid[i,j],ygrid[i,j])
        end
    end
    plot_wireframe(xgrid, ygrid, z, linewidth=0.1)

    #ax = gca(projection="3d")
    vx1=[]
    vx2=[]
    vf=[]
    for ind=1:nPop
        x1,x2 = decoderVariables(pop[ind,:])
        f = shubert(x1,x2)
        push!(vx1,x1)
        push!(vx2,x2)
        push!(vf,f)
    end
    plot3D(vx1,vx2,vf,".",c="black",markersize=3)
end

function plotpop(couleur, pop)
    vx1=[]
    vx2=[]
    vf=[]
    for ind=1:nPop
        x1,x2 = decoderVariables(pop[ind,:])
        f = shubert(x1,x2)
        push!(vx1,x1)
        push!(vx2,x2)
        push!(vf,f)
    end
    plot3D(vx1,vx2,vf,".",c=couleur,markersize=3)
end

function AlgorithmeGenetique(env)

    # Initialisations
    pop = genererPopulationInitiale(env, nPop, lVar)

    avgPop = fitnessAvgPopulation(pop)
    allAvgPop = [] # liste des fitness moyens pour chaque generation
    push!(allAvgPop, avgPop)

    minGlobal = fitnessMinPopulation(pop)
    allMinPop = [] # liste des fitness min pour chaque generation
    push!(allMinPop, minGlobal)

    println("Fitness pop init = ", avgPop)

    if env.displayGraphic == true
        initplotpop(pop)
    end

    # Generations de l'AG
    for generation = 1:nGenerations

        newPop = zeros(Int8,nPop,2*lVar) # individus de la nouvelle generation

        for couple = 1:div(nPop,2)

            # Selection des parents 1 et 2 dans la populaton par tournoi binaire
            iParent1, parent1, fparent1 = selectionTournoiPopulation(env, pop)
            iParent2, parent2, fparent2 = selectionTournoiPopulation(env, pop)
            if env.displayPrint == true
                @printf("Parents   : [%3d] %8.3f   |   [%3d] %8.3f \n", iParent1, fparent1, iParent2, fparent2)
            end

            # Operateur evolutionnaire : procreation des enfants par crossover
            enfant1, enfant2 = crossover(env, parent1,parent2)
            afficherIndividu(env, 1,enfant1) ;  afficherIndividu(env, 2,enfant2)

            # Operateur evolutionnaire : perturbation des enfants par mutation
            enfant1 = mutation(env, enfant1)
            afficherIndividu(env, 1,enfant1)

            enfant2 = mutation(env, enfant2)
            afficherIndividu(env, 2,enfant2)

            fenfant1 = evaluerFonction(enfant1)
            fenfant2 = evaluerFonction(enfant2)

            newPop[2*couple-1,:] = selectionTournoiSurvivant(env, parent1,parent2)
            newPop[2*couple,:]   = selectionTournoiSurvivant(env, enfant1,enfant2)

            # Archive la meilleure valeur trouvee jusqu'a present
            minGlobal = min(minGlobal, fparent1, fparent2, fenfant1, fenfant2)
            if env.displayPrint == true
                println(" ")
            end
        end
        pop = deepcopy(newPop)

        avgPop = fitnessAvgPopulation(pop)
        push!(allAvgPop, avgPop)
        println("Fitness pop = ", avgPop)
        push!(allMinPop, minGlobal)

        if env.displayGraphic == true
            if generation==5
                plotpop("blue", pop)
            elseif generation==10
                plotpop("orange", pop)
            elseif generation==nGenerations
                plotpop("red", pop)
            end
        end

    end
    return minGlobal, allMinPop, allAvgPop
end

# =============================================================================

# Activation ou pas des sorties ecran
displayPrint=false; displayGraphic=true
env = t_environment(displayPrint,displayGraphic)

allAvgRun = [] # liste des fitness moyens pour chaque run
if env.displayGraphic == true
    using PyPlot
    plotfct()
end

# Go pour l'experimentation numerique
for run = 1:nRun
    minGlobal, allMinPop, allAvgPop = AlgorithmeGenetique(env)
    println("Minimum global trouve = ", minGlobal)
    push!(allAvgRun, allAvgPop[end])

    if env.displayGraphic == true
        plotAvg(allMinPop, allAvgPop)
    end
end

# Resultats
println("Fitness Min / Moy / Max population moyenne sur tous les runs : ")
println(" Min = ", minimum(allAvgRun))
println(" Moy = ", mean(allAvgRun))
println(" Max = ", maximum(allAvgRun))

println("That's all folk!")
