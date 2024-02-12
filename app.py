from controller import PlantAssistantController
from view import PlantView
from MlModel import MLModel
from ChatBotModel import ChatBotModel
import dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import OpenAI, ChatOpenAI

dotenv.load_dotenv()

# if __name__ == '__main__':
#     ChatBotModel = ChatBotModel()
#     MLModel = MLModel()
#     view = PlantView()
#     controller = PlantAssistantController(MLModel, ChatBotModel, view)
#     controller.run()

text = """
cure of disease
=========================================
ID: 253
----

182   Handbook of Plant Disease Identification and Management
Contact fungicides not prone to resistance such as Captan are viable choices. Copper or Bordeaux 
mixture are traditional controls but are less effective than chemical fungicides and can cause russet -
ing of the fruit. Wettable sulfur also provides some control. The timing of application and concen -
tration varies between compounds.
Fifteen genes have been found in apple cultivars that confer resistance against apple scab. 
Researchers hope to use cisgenic techniques to introduce these genes into commercial cultivars and 
therefore create new resistant cultivars. This can be done through conventional breeding but would take over 50 years to achieve.
Tables 4.1 and 4.2 provide some information on temperature and wetting periods required for 
infection to occur and resistant, susceptible and immune varieties of apples.
4.3  FIRE BLIGHT OF APPLE
Fire blight is a common and very destructive bacterial disease of apples and pears. The disease is so named because infected leaves on very susceptible trees will suddenly turn brown, appear -
ing as though they had been scorched by fire. The disease is also referred to as blossom blight, spur blight, fruit blight, twig blight, or rootstock blight—depending on the plant part that is attacked. Economic losses to fire blight occur due to a loss of fruit-bearing surface and tree mortality. Trees may need to be removed and replanted or, in severe cases, whole blocks of trees may need to be replaced.
4.3.1  Causal  Organism
Species Associated Disease Phase Economic Importance
Erwinia amylovora Blossom Blight, Shoot Blight, and Branch and Trunk Canker. Severe
FIGURE 4.6  Disease cycle of V. inaequalis.

=========================================
=========================================
ID: 553
----

476   Handbook of Plant Disease Identification and Management
should ensure free circulation of air. Spraying Bordeaux mixture (4:4:100) once or twice on young 
bunches prevents the infection. Copper fungicides are preferred for spraying on bunches, as they do not leave any visible deposits on the fruit surface. Other than these, the most common fungicides that prove to be excellent for certain regions of the United States for controlling black rot are Sovran 50WG, Flint 50WG, Abound Flowable (2.08F), and Pristine 38WDG (Table 13.5).
13.5  ARMILLARIA ROOT ROT IN GRAPES
Armillaria root rot in grapes caused by the fungus Armillaria mellea  infects vine roots, killing 
the cambium and decaying the underlying xylem (the water-conducting system). Often found on newly cleared land, this root pathogen is native to the Pacific Northwest where it occurs on the roots of many forest tree species including Douglas-fir, madrone, oak, willow, and yellow pine. It also attacks black and red raspberries and trailing berries. The host range includes over 500 species of woody plants, making its common name of “oak root fungus” slightly misleading.
This fungus may form mushrooms at the base of infected vines in fall and winter. Mushrooms 
produce windblown spores, but these spores are not a significant means of infecting healthy vines. The fungus spreads vegetatively belowground, which leads to the formation of groups of dead and dying plants called “disease centers”. The fungus can survive on woody host roots long after the host dies. Its vegetative fungal tissue (mycelium) decomposes root wood for nutrients as it grows. When infected plants are removed, infected roots that remain below ground serve as a source of inoculum for vines planted in the same location.
FIGURE 13.27  Mycelial growth seen beneath the bark of the root.
TABLE  13.5
Common Fungicides
Fungicide Rate Comments
Sovran 50WG 3.2-4.8 oz/A Sovran is excellent for control of 
black rot
Flint 50WG 2.0 oz/A registered for the control of black 
rot
Abound 11–15.4 fl oz/A Provides good control
Pristine 38WDGCombination of two active ingredients 
(pyraclostrobin, 12.8% and boscalid 25.2%)6–10.5 oz/A
A maximum of six applications 
may be made per seasonRead the label carefully before the 
use as not prescribed for certain varieties due to foliar injury.

=========================================
=========================================
ID: 259
----

187 Apple  
4.3.5  m anagement
No single method is adequate to effectively control fire blight. A combination of practices is needed 
to reduce the severity of the disease.
 1. Choose the proper cultivars. Apple cultivars differ widely in their susceptibility to fire 
blight. During warm and rainy weather, cultivars rated moderately susceptible or moder -
ately resistant will develop shoot infections; however, the extent to which shoot infections progress will be less in resistant cultivars than in susceptible cultivars. Commercial grow -
ers should select rootstocks that are less susceptible to fire blight.
 2. Select planting sites with good soil drainage. Trees are more susceptible to fire blight in poorly drained sites than in well-drained ones. Tree productivity will also be lower on such sites. Drainage can often be improved by tiling.
 3. Follow proper pruning and fertilization practices. Using nitrogen-containing fertilizer and/or doing heavy pruning promotes vigorous growth and increases susceptibility. Fertilization and pruning practices on susceptible cultivars should be adjusted to limit excessive growth. For bearing trees, moderate shoot growth is six to twelve inches (15 to 30 cms) per year. If the growth is more than twelve inches, do not apply fertilizer until shoot growth is reduced to less than 6 inches. Apply fertilizer in the early spring (six weeks before bloom) or apply in late fall after growth has ceased. Applications in midseason prolong the time during which shoots are susceptible to infection and increase the likelihood of winter injury to tender wood.
 4. Prune out fire blight cankers during the dormant season. Delay the removal of infected shoots until the dormant season to avoid spreading infection to healthy shoots. Make pruning cuts at least six inches (15 cms) below the last point of visible infection. After each pruning cut, sterilize the pruning shears by dipping them in a freshly made solution of one part liquid bleach (Clorox, Purex, Saniclor, Sunny Sol) added to four parts water. Examine the larger branches and trunks carefully for cankers, since these are likely to overwinter and produce new 
FIGURE 4.9  Disease cycle of fire blight.

=========================================
=========================================
ID: 551
----

474   Handbook of Plant Disease Identification and Management
Traditional fungicide recommendations specified regular applications from the early shoot growth 
stage through veraison.
However, because fruit are most susceptible during the first few weeks after the start of bloom, 
this is when the fungicidal component of black rot management programs should be focused most strongly, whether additional sprays are applied or not. Sulfur is not effective for black rot control. 
FIGURE 13.23  Early symptoms of berry infection.
FIGURE 13.24  Infected berries with numerous black pycnidia.

=========================================
=========================================
ID: 547
----

470   Handbook of Plant Disease Identification and Management
inch in diameter (approximately two weeks after infection), minute black dots appear. Relatively 
small, brown circular lesions develop on infected leaves (Figure 13.17), and within a few days tiny black spherical fruiting bodies (pycnidia) protrude from them. These fungal fruiting bodies (pyc-nidia) contain thousands of summer spores (conidia). Pycnidia are often arranged in a ring pattern, just inside the margin of the lesions. (Figure 13.18). Elongated black lesions can be seen on the petiole. Lesions may also appear on young shoots, cluster stems, and tendrils. The lesions are purple to black, oval in outline, and sunken. Pycnidia also form in these lesions (Figure 13.19); they may eventually girdle these organs (Figure 13.20), causing the affected leaves to wilt (Figure 13.21). Shoot infection results in large black elliptical lesions. These lesions may contribute to breakage of shoots by wind, or in severe cases, may girdle and kill young shoots altogether.TABLE 13.3
Chemicals and their Applications
Chemical Control Rate Comment
Topsin M WSBor1-1.5 lb Apply Topsin M at 1-1.5 lb/A at first bloom (no later than5% bloom), and repeat 14 days later if severe disease conditions persist.Topsin M is also available in 70WDG and 4.5 FL formulations.
Rovral 50WPor1.5-2 lb Rovral may be applied at 1.5-2.0 lb/A four times:1. Early to midbloom;2. Prior to bunch closing;3. Beginning of fruit ripening;4. Prior to harvest if needed.Do not make more than 4 applications of Rovral per season.Do not apply within 7 days of harvest.
Vangard 75WGor10 oz Vangard is registered for use at 10 oz/A when used alone,or at 5–10 oz/A when used in a tank mix. Timing of application is approximately the 
same as for Rovral. No more than 20 oz of Vangard may be applied per acre per crop season. Vangard cannot be applied within 7 days of harvest.
Elevate 50WGor1 lb Elevate may be applied at 1 lb/A and the timing of application is approximately 
the same as Rovral and Vangard. No more than 3 lb of Elevate may be applied per acre per season. Elevate can be applied up to, and including, the day of harvest (0-day PHI).
Scala 5SCor18 fl oz Scala is registered for use at 18 fl oz alone, or at 9 fl oz when used in a tank mix. 
Timing of application is approximately the same as for Rovral.
Switch 62.5WG 11-14 oz Switch is also registered for control of sour rot (caused bya complex of organisms). Preharvest applications may be beneficial for control of 
sour rot. See the label for additional information.
TABLE 13.4Biological Control Methods
Biological Control Rate Comment
Serenade (Bacillus subtilis) Applications are recommended on a 7-10-day schedule.No maximum seasonal application rate and 0-day PHI.Moderate level of control
Trichodex (Trichoderma harzianum)Sold as wettable powder formulation that is mixed with 
water and sprayed directly onto the plants.Primary control

=========================================
Tokens: 2273
"""

chat = ChatOpenAI(temperature=0)
messages = [
    SystemMessage(
        content=f"""
        You are a plant expert and you are given docs about a plant and it's disease, you are supposed to answer user's question based on these docs
        :{text}
        """
    ),
]
res = chat(messages).content
print(res)
