import numpy as np
from numba import int32, float64, complex128, boolean
from numba.experimental import jitclass
from numba import jit
from math import gamma as GammaF
import random
import copy

### Conversion constants ###

### Parameters ###
initState   = 0 # initial state
totalTime   = 1200 # total amount of simulation time
dtF         = 2 # full timestep 
NSteps      = int(totalTime/dtF) # number of full steps in the simulation
NTraj       = 100 # number of trajectories
NStepsPrint = 200 # number of full steps that are stored/printed/outputed
NSkip       = int(NSteps/NStepsPrint) # used to enforce NStepsPrint

# System parameters

NStates = 2 # total number of states
NR = 1 # total number of nuclear DOFs
M = np.array([2000]) # mass of each nuclear DOF (for bath DOFs, this is arbitrary)



### Functions ###

def H( R ):
    R0 = R[0]
    
    A = 0.01
    B = 1.6
    C = 0.005
    D = 1.0

    H = np.zeros((2,2), dtype=np.complex128)
    if ( R0 > 0 ):
        H[0,0] = A * ( 1 - np.exp(-B*R0) )
    else:
        H[0,0] = -A * ( 1 - np.exp(B*R0) )
    
    H[1,0] = C * np.exp( -D*R0**2 )
    H[1,1] = -H[0,0]
    H[0,1] =  H[1,0]
    return H

def dH( R ): # derivative of state-dependent potential
    R0 = R[0]

    A = 0.01
    B = 1.6
    C = 0.005
    D = 1.0

    dH = np.zeros((NR,NStates,NStates), dtype=np.complex128)
    if ( R0 > 0 ):
        dH[0,0,0] = A * B * np.exp( -B*R0 )
    else:
        dH[0,0,0] = A * B * np.exp( B*R0 )

    dH[0,0,1] = -2 * C * D * R0 * np.exp(-D*R0**2)
    dH[0,1,1] = -dH[0,0,0]
    dH[0,1,0] =  dH[0,0,1]

    return dH

def dH0(sd): # derivative of state-dependent potential
    dH0 = np.zeros((NR), dtype=np.complex128)
    return dH0

def initR():
    R0 = -9.0
    P0 = 30.0
    alpha = 1.0
    sigR = 1.0/np.sqrt(2.0*alpha)
    sigP = np.sqrt(alpha/2.0)

    R = np.zeros( NR )
    P = np.zeros( NR )
    for Ri in range(NR):
        R[Ri] = random.gauss(R0, sigR )
        P[Ri] = random.gauss(P0, sigP )
    return R, P

def exact_results():
    s =\
"""
   1.0000000000000000       0.99999999999999989        0.0000000000000000        0.0000000000000000        0.0000000000000000       0.99999999999999989       1.29870769458108346E-025 -1.17083237843117973E-019  3.49047735919475508E-037
   10.000000000000000       0.99999999999999956       1.09263980930271896E-017  2.65196726159608531E-017  5.74232935978921093E-016  0.99999999999980149       1.98888391756142279E-013 -8.38060436327861296E-015 -5.74232935978921093E-016
   20.000000000000000        1.0000000000000002       1.26202653146728607E-016  3.69697945165461403E-016  3.00106195437044437E-015  0.99999999999938238       6.18462196513549970E-013 -2.27094595846670444E-014 -3.00106195437044555E-015
   30.000000000000000        1.0000000000000000       4.79473574234247420E-016  9.17853948613163984E-016  7.45007275771691533E-015  0.99999999999887357       1.12516230243048518E-012 -3.74003718947076233E-014 -7.45007275771691218E-015
   40.000000000000000       0.99999999999999878       1.25169223242118986E-015  1.85454423817236980E-015  1.60835056233051151E-014  0.99999999999825551       1.74406421668432579E-012 -5.70149649286285706E-014 -1.60835056233051214E-014
   50.000000000000000       0.99999999999999756       2.66349495225282908E-015  2.41202259468684906E-015  2.77567668594849667E-014  0.99999999999755840       2.44165374199367154E-012 -7.56679961367030841E-014 -2.77567668594849698E-014
   60.000000000000000       0.99999999999999611       4.86404368440828192E-015 -1.60525814092794706E-015  4.10552170948243880E-014  0.99999999999679856       3.20171461313309400E-012 -8.78017535005268379E-014 -4.10552170948243943E-014
   70.000000000000000       0.99999999999999178       8.06117590102779792E-015 -2.22550879398858192E-014  5.66870727675283326E-014  0.99999999999596323       4.03674649928067249E-012 -8.56328903820269786E-014 -5.66870727675283263E-014
   80.000000000000000       0.99999999999998757       1.24639693838813532E-014 -9.30630799800729799E-014  6.64089082040971216E-014  0.99999999999511879       4.88037734292848135E-012 -3.40327227000236406E-014 -6.64089082040971342E-014
   90.000000000000000       0.99999999999998057       1.81579820549389924E-014 -2.98556791351704227E-013  5.46671950495877545E-014  0.99999999999429212       5.70595546370827809E-012  1.40941736710275282E-013 -5.46671950495877672E-014
   100.00000000000000       0.99999999999997446       2.53441126448640661E-014 -8.46819630235340507E-013 -1.39381742147113509E-014  0.99999999999344391       6.55656041561507971E-012  6.13164836772026283E-013  1.39381742147113052E-014
   110.00000000000000       0.99999999999996547       3.42229894652509734E-014 -2.23275498200865468E-012 -2.37196392189953346E-013  0.99999999999256650       7.43241736052998500E-012  1.78921959816506975E-012  2.37196392189953296E-013
   120.00000000000000       0.99999999999995492       4.48839972913074882E-014 -5.57302581071226533E-012 -8.26907037127133762E-013  0.99999999999168188       8.31612444135902298E-012  4.50078031111958168E-012  8.26907037127133762E-013
   130.00000000000000       0.99999999999994293       5.76036701621774682E-014 -1.32748571215673877E-011 -2.22178650780945938E-012  0.99999999999084999       9.14877398020812409E-012  1.03229430879832522E-011  2.22178650780946019E-012
   140.00000000000000       0.99999999999992772       7.27195966286657491E-014 -3.02857924422591394E-011 -5.28125022374827103E-012  0.99999999999014377       9.85683149530966638E-012  2.18859155457672917E-011  5.28125022374827183E-012
   150.00000000000000       0.99999999999990941       9.05500017197789259E-014 -6.63038014224724522E-011 -1.14664689368561509E-011  0.99999999998957401       1.04266526536537760E-011  4.25548179538554131E-011  1.14664689368561525E-011
   160.00000000000000       0.99999999999988864       1.11672629071164069E-013 -1.39481314368159132E-010 -2.27820899566779429E-011  0.99999999998912736       1.08729704537444359E-011  7.36426508816115092E-011  2.27820899566779493E-011
   170.00000000000000       0.99999999999986289       1.37349396350285138E-013 -2.82254863129450409E-010 -4.05513099423318031E-011  0.99999999998876610       1.12332425048523939E-011  1.04012694968879975E-010  4.05513099423318031E-011
   180.00000000000000       0.99999999999982769       1.71889889808835420E-013 -5.50108721300099167E-010 -6.04172279033159496E-011  0.99999999998843370       1.15662987287173498E-011  7.93289541405351970E-011  6.04172279033159625E-011
   190.00000000000000       0.99999999999976896       2.30872125653020945E-013 -1.03442706447379336E-009 -5.70954063768625689E-011  0.99999999998795264       1.20467505074255919E-011 -1.78667372552355926E-010  5.70954063768625689E-011
   200.00000000000000       0.99999999999963207       3.67742231097859889E-013 -1.88166267704514442E-009  5.65937465300174567E-011  0.99999999998671585       1.32838814457182140E-011 -1.16880619488438220E-009 -5.65937465300173792E-011
   210.00000000000000       0.99999999999923950       7.60685119416987648E-013 -3.32442046799542850E-009  5.31251976703295586E-010  0.99999999998281541       1.71865158698633734E-011 -4.16397943622992781E-009 -5.31251976703295689E-010
   220.00000000000000       0.99999999999801503       1.98486135464405415E-012 -5.73886376345109385E-009  2.01819647842992430E-009  0.99999999997045097       2.95482654375278768E-011 -1.22127724540229658E-008 -2.01819647842992512E-009
   230.00000000000000       0.99999999999416356       5.83691286781584758E-012 -9.76332355096504437E-009  6.09954245752194929E-009  0.99999999993237609       6.76225044026066590E-011 -3.22733855007347442E-008 -6.09954245752194929E-009
   240.00000000000000       0.99999999998224853       1.77510071827423757E-011 -1.65570196643031980E-008  1.64284831284903271E-008  0.99999999981872723       1.81270838193474925E-010 -7.96166195905987767E-008 -1.64284831284903271E-008
   250.00000000000000       0.99999999994635302       5.36467345932971477E-011 -2.83662633441496173E-008  4.11019840980227104E-008  0.99999999948937934       5.10619372313515116E-010 -1.86643691890346934E-007 -4.11019840980227104E-008
   260.00000000000000       0.99999999984130339       1.58696389378820449E-010 -4.97345807929177131E-008  9.74492543173397120E-008  0.99999999856274746       1.43725382073100415E-009 -4.20043544506691495E-007 -9.74492543173397120E-008
   270.00000000000000       0.99999999954295260       4.57046922242591144E-010 -9.00039276765677417E-008  2.21417641790242617E-007  0.99999999603291456       3.96708520689256012E-009 -9.13233865584147613E-007 -2.21417641790242617E-007
   280.00000000000000       0.99999999872086320       1.27913697672512865E-009 -1.68305132716294338E-007  4.85450650041551914E-007  0.99999998933176470       1.06682352952503228E-008 -1.92595038664492462E-006 -4.85450650041551914E-007
   290.00000000000000       0.99999999652330984       3.47669003312655272E-009 -3.23179191972829876E-007  1.03158114365594347E-006  0.99999997210892644       2.78910745242486875E-008 -3.95051900643136448E-006 -1.03158114365594368E-006
   300.00000000000000       0.99999999082451252       9.17548763738334189E-009 -6.30529333907679379E-007  2.13094175744335141E-006  0.99999992915457769       7.08454231231287154E-008 -7.89572966448097632E-006 -2.13094175744335183E-006
   310.00000000000000       0.99999997648748151       2.35125184859701472E-008 -1.23607531740315745E-006  4.28767095718773558E-006  0.99999982517971220       1.74820287871432322E-007 -1.53952176561840688E-005 -4.28767095718773558E-006
   320.00000000000000       0.99999994149443250       5.85055677466689320E-008 -2.41222530761874229E-006  8.41493668833724849E-006  0.99999958085891549       4.19141083946629772E-007 -2.93082882262648034E-005 -8.41493668833724849E-006
   330.00000000000000       0.99999985862967244       1.41370326867725187E-007 -4.65465472851082755E-006  1.61240098958399697E-005  0.99999902339804370       9.76601957134125657E-007 -5.45066044635124158E-005 -1.61240098958399731E-005
   340.00000000000000       0.99999966823870301       3.31761296399502709E-007 -8.84110458961154716E-006  3.01840003926360624E-005  0.99999778796578886       2.21203420934302045E-006 -9.90676945486612112E-005 -3.01840003926360658E-005
   350.00000000000000       0.99999924377760796       7.56222392251817468E-007 -1.64838208108340771E-005  5.52290530304858483E-005  0.99999512773020138       4.87226979945407759E-006 -1.76020723504901998E-004 -5.52290530304858348E-005
   360.00000000000000       0.99999832549785805       1.67450214237980000E-006 -3.01167347326794934E-005  9.88089865151602992E-005  0.99998955999130756       1.04400086907938104E-005 -3.05802850213408558E-004 -9.88089865151603128E-005
   370.00000000000000       0.99999639756435810       3.60243564104285396E-006 -5.38667717093944484E-005  1.72893936208402089E-004  0.99997822903564626       2.17709643520138624E-005 -5.19573416547918290E-004 -1.72893936208402008E-004
   380.00000000000000       0.99999246905979011       7.53094020978733050E-006 -9.42616195296955408E-005  2.95946513251757069E-004  0.99995579753372299       4.42024662760863329E-005 -8.63482591961948587E-004 -2.95946513251757069E-004
   390.00000000000000       0.99998469902812814       1.53009718715553256E-005 -1.61317718773395797E-004  4.95657088594906301E-004  0.99991258354830936       8.74164516894955848E-005 -1.40388369114796793E-003 -4.95657088594906301E-004
   400.00000000000000       0.99996978054643493       3.02194535652804908E-005 -2.69923712775744118E-004  8.12388689447575895E-004  0.99983154013164899       1.68459868351171087E-004 -2.23329819054597124E-003 -8.12388689447576004E-004
   410.00000000000000       0.99994197172825261       5.80282717480780271E-005 -4.41476157233089558E-004  1.30328829752247226E-003  0.99968353989600067       3.16460103999020566E-004 -3.47667858247545586E-003 -1.30328829752247204E-003
   420.00000000000000       0.99989163907004186       1.08360929958111263E-004 -7.05626468233519120E-004  2.04688585281171190E-003  0.99942030725427622       5.79692745721681378E-004 -5.29716527265260920E-003 -2.04688585281171103E-003
   430.00000000000000       0.99980317165103383       1.96828348966154279E-004 -1.10185554943763135E-003  3.14782360590788431E-003  0.99896429458828728       1.03570541171160496E-003 -7.90011331759988764E-003 -3.14782360590788474E-003
   440.00000000000000       0.99965214875527963       3.47851244721144974E-004 -1.68040942503606678E-003  4.74115031250642218E-003  0.99819491968599483       1.80508031400539756E-003 -1.15337087841356056E-002 -4.74115031250642131E-003
   450.00000000000000       0.99940171627814667       5.98283721853903926E-004 -2.50192545351314620E-003  6.99540344668030979E-003  0.99693099901091764       3.06900098908121158E-003 -1.64840694043818549E-002 -6.99540344668031066E-003
   460.00000000000000       0.99899826450809104       1.00173549190944299E-003 -3.63489572242366513E-003  1.01135257414230358E-002  0.99491006321711395       5.08993678288489036E-003 -2.30624411039361543E-002 -1.01135257414230358E-002
   470.00000000000000       0.99836670813401718       1.63329186598259123E-003 -5.15001698675056959E-003  1.43305640943160277E-002  0.99176662027083906       8.23337972915886944E-003 -3.15821267404457484E-002 -1.43305640943160277E-002
   480.00000000000000       0.99740593762773855       2.59406237226147366E-003 -7.11054833931097727E-003  1.99071222378332420E-002  0.98701328680707390       1.29867131929246957E-002 -4.23233344218705565E-002 -1.99071222378332455E-002
   490.00000000000000       0.99598530344497427       4.01469655502684382E-003 -9.55812459977310562E-003  2.71177167194844235E-002  0.98003075156261610       1.99692484373797802E-002 -5.54854434036507610E-002 -2.71177167194844235E-002
   500.00000000000000       0.99394324682657087       6.05675317342956390E-003 -1.24941165586001379E-002  3.62335344784614349E-002  0.97007416087234388       2.99258391276523948E-002 -7.11284259973690575E-002 -3.62335344784614349E-002
   510.00000000000000       0.99108931702178182       8.91068297821773735E-003 -1.58575904226348798E-002  4.74996040169256561E-002  0.95630385152477682       4.36961484752174251E-002 -8.91082993364671339E-002 -4.74996040169256561E-002
   520.00000000000000       0.98721072199265436       1.27892780073470443E-002 -1.95021070339105318E-002  6.11070434627021680E-002  0.93784647250862530       6.21535274913699448E-002 -0.10901512665254245      -6.11070434627021680E-002
   530.00000000000000       0.98208417555234406       1.79158244476561070E-002 -2.31748163201324769E-002  7.71617905528649978E-002  0.91388783979845745       8.61121602015400001E-002 -0.13012543039247484      -7.71617905528649978E-002
   540.00000000000000       0.97549310674705425       2.45068932529451805E-002 -2.65022492808383060E-002  9.56519865961512061E-002  0.88379154182477737       0.11620845817522234      -0.15138273939369376      -9.56519865961512200E-002
   550.00000000000000       0.96724934269535467       3.27506573046456237E-002 -2.89875551157945943E-002  0.11641689550641528       0.84722866251521567       0.15277133748478539      -0.17141912699023518      -0.11641689550641526     
   560.00000000000000       0.95721730776569258       4.27826922343070778E-002 -3.00233929356998955E-002  0.13912078811951700       0.80429635352516704       0.19570364647483249      -0.18862618061917111      -0.13912078811951697     
   570.00000000000000       0.94533781779126258       5.46621822087377321E-002 -2.89231380580326075E-002  0.16323549434298659       0.75559918887755806       0.24440081112244158      -0.20127596332266531      -0.16323549434298659     
   580.00000000000000       0.93164793832942794       6.83520616705716177E-002 -2.49706085452357437E-002  0.18803520328004478       0.70226959436008840       0.29773040563991043      -0.20768248985694887      -0.18803520328004478     
   590.00000000000000       0.91629334045762811       8.37066595423720805E-002 -1.74855196124132535E-002  0.21260648241196237       0.64591286435477846       0.35408713564522204      -0.20638442070037302      -0.21260648241196234     
   600.00000000000000       0.89953024669169124       0.10046975330830846      -5.89888907542795559E-003  0.23587536087050218       0.58847686876926641       0.41152313123073270      -0.19632285122882562      -0.23587536087050218     
   610.00000000000000       0.88171538206029143       0.11828461793970867       1.01697289740848313E-002  0.25665174178251970       0.53206296258877483       0.46793703741122539      -0.17698652793178801      -0.25665174178251970     
   620.00000000000000       0.86328413220019051       0.13671586779981007       3.08424847404841114E-002  0.27368954431843040       0.47870823815775027       0.52129176184225023      -0.14850141440642137      -0.27368954431843040     
   630.00000000000000       0.84471902629197648       0.15528097370802407       5.59299769919983647E-002  0.28575908540496298       0.43017610443797860       0.56982389556202140      -0.11165127817444202      -0.28575908540496298     
   640.00000000000000       0.82651231128119984       0.17348768871879977       8.49018354804678549E-002  0.29172659074109647       0.38779022172047933       0.61220977827951972      -6.78283386656993309E-002 -0.29172659074109647     
   650.00000000000000       0.80912740691572238       0.19087259308427765       0.11688569016025335       0.29063464212121204       0.35233676666915259       0.64766323333084519      -1.89247708270355273E-002 -0.29063464212121204     
   660.00000000000000       0.79296420323862238       0.20703579676137682       0.15069573143595577       0.28177699702759013       0.32404491304669919       0.67595508695329320       3.28158513880068697E-002 -0.28177699702759013     
   670.00000000000000       0.77833245090018544       0.22166754909981379       0.18488875124445711       0.26476159828861096       0.30263962805617400       0.69736037194381317       8.49648367077491251E-002 -0.26476159828861096     
   680.00000000000000       0.76543606770029904       0.23456393229970124       0.21784277424466750       0.23955663344561318       0.28744840487042128       0.71255159512954980       0.13507175162322815      -0.23955663344561323     
   690.00000000000000       0.75436936209151650       0.24563063790848308       0.24785146714805462       0.20651601255072366       0.27753690689563476       0.72246309310430001       0.18080951671456472      -0.20651601255072366     
   700.00000000000000       0.74512434136629524       0.25487565863370448       0.27322656708357973       0.16638237115518090       0.27184821113557245       0.72815178886429044       0.22009257061503593      -0.16638237115518090     
   710.00000000000000       0.73760678163698512       0.26239321836301455       0.29240052901771507       0.12026744453723556       0.26932513774841971       0.73067486225130052       0.25116565850298989      -0.12026744453723562     
   720.00000000000000       0.73165782814601699       0.26834217185398368       0.30402226978548835       6.96112227078062307E-002  0.26900279505465863       0.73099720494479259       0.27266520796189869      -6.96112227078062723E-002
   730.00000000000000       0.72707765273334868       0.27292234726665165       0.30704004546574881       1.61225744656340632E-002  0.27006656527285322       0.72993343472610905       0.28365742093159685      -1.61225744656340354E-002
   740.00000000000000       0.72364805383870345       0.27635194616129716       0.30076691273226958      -3.82950219062757960E-002  0.27187747865068679       0.72812252134742161       0.28365744740985044       3.82950219062757960E-002
   750.00000000000000       0.72115166602276570       0.27884833397723485       0.28492571115580967      -9.16283812514800539E-002  0.27397131469319580       0.72602868530349018       0.27263303687282142       9.16283812514800955E-002
   760.00000000000000       0.71938642354345705       0.28061357645654350       0.25967193623379070      -0.14184556250894054       0.27603969913209875       0.72396030086231489       0.25099467707117701       0.14184556250894054     
   770.00000000000000       0.71817487957392767       0.28182512042607233       0.22559417754413844      -0.18698821654757711       0.27790138456623464       0.72209861542470155       0.21957304287455187       0.18698821654757711     
   780.00000000000000       0.71736875802802869       0.28263124197197148       0.18369293625188343      -0.22526083216835024       0.27947052322731475       0.72052947675853074       0.17958392002258361       0.22526083216835027     
   790.00000000000000       0.71684962362512294       0.28315037637487706       0.13533959842343238      -0.25511271849132866       0.28072677346579655       0.71927322651292336       0.13258070782802064       0.25511271849132861     
   800.00000000000000       0.71652678662441793       0.28347321337558129       8.22181246373951541E-002 -0.27530905125223459       0.28169008581820787       0.71830991415099255       8.03950249287781055E-002  0.27530905125223459     
   810.00000000000000       0.71633355576042468       0.28366644423957527       2.62526265710926465E-002 -0.28498792216287061       0.28240133474366647       0.71759866521340809       2.50666458706797733E-002  0.28498792216287061     
   820.00000000000000       0.71622279017380719       0.28377720982619331      -3.04755568305034377E-002 -0.28370103454509615       0.28290875186201020       0.71709124808037328      -3.12352175490062806E-002  0.28370103454509615     
   830.00000000000000       0.71616245776551180       0.28383754223448876      -8.58173917210854065E-002 -0.27143645464978428       0.28325938418333485       0.71674061574217118      -8.62965672113501781E-002  0.27143645464978439     
   840.00000000000000       0.71613165035492876       0.28386834964507202      -0.13764827596247212      -0.24862262608094923       0.28349447021091395       0.71650552969629211      -0.13794594814689279       0.24862262608094923     
   850.00000000000000       0.71611728101378525       0.28388271898621520      -0.18395877354349496      -0.21611365491244072       0.28364758587351907       0.71635241401509708      -0.18414089791554494       0.21611365491244072     
   860.00000000000000       0.71611151922523486       0.28388848077476614      -0.22294014038258703      -0.17515664723867272       0.28374454815436012       0.71625545171677940      -0.22304988592739111       0.17515664723867266     
   870.00000000000000       0.71610990982821510       0.28389009017178390      -0.25306112699955063      -0.12734260296653346       0.28380428273494895       0.71619571712133179      -0.25312625786644932       0.12734260296653341     
   880.00000000000000       0.71611006460154925       0.28388993539844953      -0.27313296453212305      -7.45430161008197867E-002  0.28384009510509539       0.71615990474034363      -0.27317103190103775       7.45430161008198144E-002
   890.00000000000000       0.71611079749735329       0.28388920250264660      -0.28235996497572552      -1.88348819650326280E-002  0.28386099133211329       0.71613900850756684      -0.28238187615843074       1.88348819650326280E-002
   900.00000000000000       0.71611158167688405       0.28388841832311529      -0.28037377576963440       3.75827516390008151E-002  0.28387285608018514       0.71612714375938047      -0.28038619530705877      -3.75827516390008151E-002
   910.00000000000000       0.71611222644403039       0.28388777355596906      -0.26725000162529022       9.24772430964614228E-002  0.28387940809132522       0.71612059175374043      -0.26725693352760538      -9.24772430964614228E-002
   920.00000000000000       0.71611269610041317       0.28388730389958589      -0.24350661912210309       0.14367220691762586       0.28388292308355295       0.71611707677201353      -0.24351042882056889      -0.14367220691762583     
   930.00000000000000       0.71611301527702509       0.28388698472297508      -0.21008433666918069       0.18913510604389985       0.28388475085634940       0.71611524901363843      -0.21008639828720618      -0.18913510604389985     
   940.00000000000000       0.71611322382982767       0.28388677617017299      -0.16830976759256497       0.22705904939531271       0.28388566783435593       0.71611433205260144      -0.16831086609096993      -0.22705904939531274     
   950.00000000000000       0.71611335823039945       0.28388664176960060      -0.11984296101113215       0.25593550654118413       0.28388610731413338       0.71611389259089619      -0.11984353733402467      -0.25593550654118413     
   960.00000000000000       0.71611344593819570       0.28388655406180369      -6.66114485112820065E-002  0.27461502765533646       0.28388630399902109       0.71611369592385832      -6.66117462383466474E-002 -0.27461502765533646     
   970.00000000000000       0.71611350538169727       0.28388649461830262      -1.07334912237422760E-002  0.28235354570045418       0.28388638139118177       0.71611361854826627      -1.07336426772032430E-002 -0.28235354570045418     
   980.00000000000000       0.71611354785537740       0.28388645214462271       4.55663683536978170E-002  0.27884241997091685       0.28388640274460714       0.71611359720940582       4.55662924810652237E-002 -0.27884241997091685     
   990.00000000000000       0.71611357968620215       0.28388642031379691       0.10004604867589217       0.26422103296443167       0.28388639971269580       0.71611360025350890       0.10004601124008292      -0.26422103296443167     
   1000.0000000000000       0.71611360406606261       0.28388639593393700       0.15053540214272768       0.23907145053015377       0.28388638786117820       0.71611361211478197       0.15053538394823412      -0.23907145053015377     
   1010.0000000000000       0.71611362241922039       0.28388637758077995       0.19502283360062864       0.20439537104633915       0.28388637473668932       0.71611362524675204       0.19502282488998837      -0.20439537104633915     
   1020.0000000000000       0.71611363535059891       0.28388636464940087       0.23173559895525547       0.16157429498307954       0.28388636388202215       0.71611363610692780       0.23173559484940809      -0.16157429498307954     
   1030.0000000000000       0.71611364326147509       0.28388635673852436       0.25921057500395100       0.11231451394136037       0.28388635674512763       0.71611364324772597       0.25921057310270651      -0.11231451394136034     
   1040.0000000000000       0.71611364671010125       0.28388635328989764       0.27635267477980818       5.85791219023091628E-002  0.28388635355964281       0.71611364643587594       0.27635267392122870      -5.85791219023091628E-002
   1050.0000000000000       0.71611364657623167       0.28388635342376750       0.28247857751368038       2.50976719732675735E-003  0.28388635376953131       0.71611364622774354       0.28247857714395691      -2.50976719732675735E-003
   1060.0000000000000       0.71611364407601619       0.28388635592398320       0.27734402930018304      -5.36587287602721258E-002  0.28388635628368042       0.71611364371471153       0.27734402915912626       5.36587287602721258E-002
   1070.0000000000000       0.71611364066935257       0.28388635933064743       0.26115362653982005      -0.10768747683745770       0.28388635968701237       0.71611364031206681       0.26115362650655621       0.10768747683745770     
   1080.0000000000000       0.71611363790247018       0.28388636209752882       0.23455269325491032      -0.15742279311362922       0.28388636244765808       0.71611363755182889       0.23455269327350467       0.15742279311362922     
   1090.0000000000000       0.71611363722876609       0.28388636277123414       0.19860157768593897      -0.20088207713411793       0.28388636311591009       0.71611363688381358       0.19860157773013146       0.20088207713411793     
   1100.0000000000000       0.71611363984683862       0.28388636015316104       0.15473339473355716      -0.23633286944378318       0.28388636049361188       0.71611363950624241       0.15473339479042794       0.23633286944378318     
   1110.0000000000000       0.71611364658534149       0.28388635341465845       0.10469690086898711      -0.26236193508788463       0.28388635375184529       0.71611364624808049       0.10469690093147874       0.26236193508788463     
   1120.0000000000000       0.71611365785066727       0.28388634214933406       5.04867807757734890E-002 -0.27793161812210671       0.28388634248389066       0.71611365751607381       5.04867808393965958E-002  0.27793161812210671     
   1130.0000000000000       0.71611367363910705       0.28388632636089306      -5.73587339879484334E-003 -0.28242122023070454       0.28388632669323083       0.71611367330675146      -5.73587333739033686E-003  0.28242122023070454     
   1140.0000000000000       0.71611369360325317       0.28388630639674739      -6.17297186621945451E-002 -0.27565175401616726       0.28388630672717780       0.71611369327281438      -6.17297186058057071E-002  0.27565175401616726     
   1150.0000000000000       0.71611371715456396       0.28388628284543593      -0.11526251798043503      -0.25789308455983034       0.28388628317422104       0.71611371682577496      -0.11526251793146598       0.25789308455983034     
   1160.0000000000000       0.71611374358145008       0.28388625641855036      -0.16420013435672193      -0.22985317504586020       0.28388625674589707       0.71611374325410171      -0.16420013431718103       0.22985317504586020     
   1170.0000000000000       0.71611377216334127       0.28388622783665868      -0.20659161400980222      -0.19264986564246550       0.28388622816274711       0.71611377183725211      -0.20659161398133677       0.19264986564246550     
   1180.0000000000000       0.71611380226394794       0.28388619773605150      -0.24074696656277705      -0.14776631106149668       0.28388619806100501       0.71611380193899410      -0.24074696654661792       0.14776631106149668     
   1190.0000000000000       0.71611383339106216       0.28388616660893828      -0.26530454137388776      -9.69918535691417921E-002  0.28388616693284613       0.71611383306715415      -0.26530454137073600       9.69918535691417921E-002
   1200.0000000000000       0.71611386521477338       0.28388613478522612      -0.27928531391502992      -4.23506887688689382E-002  0.28388613510816646       0.71611386489183304      -0.27928531392505157       4.23506887688689382E-002
"""

    s = np.array([float(x) for x in s.split()]).reshape(-1, 9)
    return s
        


