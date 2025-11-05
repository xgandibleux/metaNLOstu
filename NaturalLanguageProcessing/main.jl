# Cours      : Metaheuristiques
# Algorithme : Algorithmes genetiques - Extraction automatique de resume
# Auteur     : Xavier Gandibleux
# Date       : 2025 


using TextAnalysis
using Random
using LinearAlgebra
using Statistics

# =============================================================================
# 0) Texte d'entree à resumer (tire de la these de doctorat de Manon Perrignon)

text = """
D’ici 2050, la population mondiale devrait atteindre environ 9,7 milliards d’habitants tirant la demande alimentaire vers des niveaux de production que le secteur aura du mal à satisfaire en respectant les préceptes de la durabilité sans mutations profondes de la filière. 
Les industries agroalimentaires opèrent dans un environnement compétitif et aux ressources de plus en plus limitées, leur imposant de se démarquer par de l’innovation produit, procédé ou organisationnelle; quand elles y parviennent, leur productivité accrue leur permet de concilier rentabilité et prix abordables pour les consommateurs. 
Bien qu’elles génèrent de nombreux bénéfices sociaux et économiques (emploi de millions de personnes, création de valeur, etc.), les industries agroalimentaires ont un impact environnemental significatif. 
L’empreinte carbone du secteur englobe, en effet, les émissions de gaz à effet de serre générées tout au long de la chaîne de production alimentaire : des matières premières agricoles jusqu’à leur consommation finale, en passant par la transformation, le transport, la distribution et la préparation. 
Ce périmètre du « champ à l’assiette » permet d’évaluer l’impact environnemental complet de l’alimentation. 
En France, le secteur alimentaire représente 22% de l’empreinte carbone moyenne d’un citoyen. 
Au sein de cette chaîne, la répartition des émissions indique que l’agriculture constitue la principale source d’impact avec 53% des émissions, tandis que la transformation alimentaire représente 7% de l’empreinte carbone totale du secteur. 
Ainsi, la combinaison de modes de production inchangés et d’une demande alimentaire croissante ne peut qu’amplifier les effets néfastes sur l’environnement, rendant impérative une plus grande optimisation des procédés de transformation, de l’utilisation des matières premières agricoles, de l’eau et de l’énergie, tout en garantissant la sécurité des aliments et leur qualité.
Face à ces défis, il apparaît nécessaire de redéfinir la notion de performance industrielle dans le secteur agroalimentaire. 
Celle-ci se matérialise avant tout dans la manière dont les procédés de transformation sont conçus et pilotés, véritables leviers de compétitivité et de création de valeur pour l’entreprise. 
Ces procédés se caractérisent par l’ensemble des opérations unitaires, physiques, chimiques, biologiques ou mécaniques, mises en œuvre de manière séquentielle ou simultanée pour transformer des matières premières agricoles en produits alimentaires sûrs, stables et adaptés à la consommation humaine. 
Ils peuvent inclure des étapes telles que la fermentation, la cuisson, la pasteurisation, le séchage, l’extrusion, ou encore l’emballage.
Les procédés agroalimentaires sont définis par plusieurs étapes dépendantes les unes des autres, impliquent des interactions entre plusieurs phénomènes (par exemple des évolutions microbiologiques et physico-chimiques lors de la fermentation) et peuvent être décrits par plusieurs indicateurs de performances parfois contradictoires, ce qui les rend complexes. 
Dans ce contexte, l’évaluation de la performance globale des procédés de transformation agroalimentaires repose sur trois dimensions majeures : la dimension économique, liée à la productivité, aux coûts et à la rentabilité ; la dimension environnementale, qui concerne l’usage des ressources, de l’énergie et de l’eau ainsi que les émissions associées ; et la dimension qualitative, qui recouvre la sécurité, les caractéristiques organoleptiques et la valeur nutritionnelle des produits. 
L’intégration simultanée de ces dimensions est indispensable pour garantir la durabilité et la compétitivité des entreprises. 
Limiter l’analyse à une seule dimension revient à compromettre l’optimisation globale des procédés et, à terme, la performance durable des industries agroalimentaires.
Par ailleurs, ces dimensions sont souvent contradictoires : prolonger la durée de conservation des aliments (réduction du gaspillage) peut nécessiter une intensification des traitements thermiques (augmentation de la consommation énergétique) ou encore l’ajout de conservateurs (impact négatif sur la qualité du produit). 
Ainsi, la recherche d’un compromis équilibré entre objectifs économiques, environnementaux et qualitatifs constitue un enjeu majeur pour l’optimisation et le pilotage des procédés agroalimentaires.
Récemment en industrie, l’optimisation multi-objectifs a gagné en popularité en raison de sa capacité à évaluer les problèmes selon diverses perspectives, en traitant simultanément plusieurs objectifs. 
Les problèmes multi-objectifs sont nombreux, se retrouvent dans divers domaines et le choix de la méthode d’optimisation, parmi les très nombreuses existantes, dépend de la nature du problème à résoudre. 
Dans l’industrie chimique, l’optimisation multi-objectifs est largement utilisée pour optimiser les performances de procédés . 
A l’inverse, et malgré des avancées significatives dans de nombreux secteurs, l’industrie agroalimentaire n’a pas encore pleinement exploité les avantages de ces méthodes, qui restent largement sous-utilisées. 
Récemment, les outils et méthodes d’optimisation multi-objectifs ont fait progressivement leur entrée dans le domaine des procédés alimentaires, ouvrant ainsi de nouvelles perspectives pour le secteur, qui dispose d’un atout considérable avec la richesse des données collectées tout au long des transformations. 
L’avènement de l’Industrie 4.0 et l’abondance de capteurs mesurant les paramètres des procédés pour permettre un suivi plus précis constituent un levier puissant pour optimiser la production alimentaire . 
Malgré la collecte de données étendue, leur analyse reste encore souvent imparfaite et incomplète à ce jour. 
Pourtant ces données peuvent aider dans la compréhension et le contrôle des procédés de transformation, notamment depuis l’émergence de l’intelligence artificielle et des méthodes d’apprentissage automatique, qui offrent des outils puissants pour analyser et valoriser ces volumes importants d’informations dans une approche basée sur les données.
À partir de ces constats, l’utilisation d’une méthodologie systémique englobant l’ensemble du procédé et des données collectées serait bénéfique pour optimiser la performance industrielle dans le secteur agroalimentaire. 
L’objectif réside donc dans la conception et la mise en œuvre d’une approche basée sur les données favorisant à la fois la compréhension approfondie des procédés de transformation et leur optimisation pour atteindre les objectifs de performance de l’usine.
"""


# =============================================================================
# 1) decoupage

# decoupage en phrases (expression réguliere simple)
sentences = split(strip(text), r"(?<=[\.\?!])\s+")
# retirer d'eventuels elements vides
sentences = [s for s in sentences if !isempty(s)]


# =============================================================================
# 2) nettoyage simple au niveau de la chaine  

function clean_string(s::AbstractString)
    s2 = replace(s, r"[[:punct:]]+" => " ")   # enleve la ponctuation
    s2 = lowercase(s2)                        # minuscules
    s2 = strip(s2)
    return s2
end
cleaned = [clean_string(s) for s in sentences]


# =============================================================================
# 3) construire un Corpus de TokenDocument  

crps = Corpus(TokenDocument.(cleaned))   # chaque document est TokenDocument{String}

# construire la DTM puis TF-IDF (voir : https://cuik.io/blog/lexique-seo/tf-idf-term-frequency-inverse-document-frequency/)
update_lexicon!(crps)
dtm = DocumentTermMatrix(crps)           # docs x termes
tfidf_sparse = tf_idf(dtm)               # tf-idf sparse
tfidf = Array(tfidf_sparse)              # dense matrix (docs x terms)

# vecteur "texte global" = moyenne des phrases
text_vector = vec(mean(tfidf, dims=1))


# =============================================================================
# Similarite cosinus (voir : https://medium.com/@santannalouis208/la-similarit%C3%A9-cosinus-en-ia-nlp-d554d3b14efa)

epsval() = eps(Float64)
function cosine_similarity(a::AbstractVector, b::AbstractVector)
    na = norm(a); nb = norm(b)
    return (na == 0.0 || nb == 0.0) ? 0.0 : dot(a,b) / (na*nb + epsval())
end


# =============================================================================
# Fitness
#
#   sim : 
#       Representativite : similarite entre resume et texte original
#   redundancy_penalty : 
#       Coherence / Redondance faible : penaliser si les phrases selectionnées sont trop similaires
#   length_penalty : 
#       Longueur cible : penaliser si trop court ou trop long (par ex. max 30% du texte original).

# Parametres du fitness
#   α : importance de la similarite :
#       α=0.0   →  pas de similarite
#       α=1.0   →  similarite maximale (valeur par defaut)
#   β : importance de la penalite de redondance : 
#       β=0.0   →  pas de pénalite
#       β grand →  plus forte incitation à choisir des phrases différentes
#   γ : importance de la penalite de longueur 
#       γ petit →  resume plus court
#       γ grand →  resume plus long

function fitness(chromosome::Vector{Int}; α::Float64=1.0, β::Float64=0.5, γ::Float64=0.3)    
    idx = findall(x -> x == 1, chromosome)
    if isempty(idx)
        return -Inf
    end
    chosen_matrix = tfidf[idx, :]                              # lignes = phrases choisies
    chosen_vec = vec(mean(chosen_matrix, dims=1))
    sim = cosine_similarity(chosen_vec, text_vector)

    # pénalité de longueur    
    ratio = length(idx) / length(sentences)
    length_penalty = abs(ratio - γ)                            # si γ = 0.3 => vise ~30% des phrases

    # pénalité de redondance (moyenne similarités entre phrases choisies)
    redundancies = Float64[]
    for i in 1:length(idx)-1
        for j in i+1:length(idx)
            push!(redundancies,
                cosine_similarity(tfidf[idx[i], :], tfidf[idx[j], :]))
        end
    end
    redundancy_penalty = isempty(redundancies) ? 0.0 : mean(redundancies)

    return α*sim - length_penalty - β*redundancy_penalty    
end


# =============================================================================
# Algorithme genetique
#
#    Chaque individu de la population represente un resume candidat.
#    Codage d'un individu : vecteur binaire de longueur = nb de phrases du texte (1 = phrase incluse, 0 = exclue).
#    Exemple : [1, 0, 0, 1, 1] → resume compose des phrases 1, 4 et 5.

function genetic_algorithm(; pop_size::Int=40, 
                             generations::Int=100, 
                             mutation_rate::Float64=0.05, 
                             α::Float64=1.0, 
                             β::Float64=0.5, 
                             γ::Float64=0.3
                            )    


    # a mettre en place...

end


# =============================================================================
# Point d'entree principal

# a mettre en place...

# exemples d'appels utiles
resume = rand(0:1, length(cleaned))   # une solution aleatoire (un resume)
evaluation = fitness(resume)          # mesure de fitness d'une solution
