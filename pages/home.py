from dash import html

home_layout = html.Div([
    html.H1("Duurzame Polder Decision Support System (DSS) Home Page"),

    html.P("This is the home page of the DSS. Here you can read about project background and the main functions of this tool. "),

    html.H2("Duurzame Polder - Sustainable Energy for cities of Den Bosch and Oss"),
    html.P('''
                Duurzame Polder is a sustainable energy project initiated by the municipalities of Den Bosch and Oss,
                which are both part of the Noord-Brabant Province of the Netherlands. It started from a province-wide
                search for a location to put a large sustainable energy project to move towards sustainable energy generation
                goals. It was decided that the most fitting option is the polder located between den Bosch and Oss, with
                municipality borders going through it. In order to realize this project, the Duurzame Polder organization
                was established. The project has been going through various exploration and assessment phases since 2017
                and is now in the stage of getting its first permits and official approvals. Here is the project website (in Dutch):
                https://www.duurzamepolder.nl/
                '''),
    html.Div(children = [html.Img(src = 'assets/polderPlan.png', style={'height':'40%', 'width':'40%'}),
                         html.Img(src = 'assets/prefAltPlan.png', style={'height':'40%', 'width':'40%'})],style={"textAlign": "center"}),
    html.P('''
                What is interesting about this project is that the alternative they ended up going with as the "Preferred Alternative"
                makes nobody happy. It performs badly on "sound pollution" and "effect on bird habitats" parameter, while barely 
                accomplishing the energy goals. This can well and fully be considered a large, complex project for the
                municipalities Den Bosch and Oss. For this reason, it was decided that this case will be used as a basis for
                the proposed DSS, which aims to improve the performance of large complex projects by intercepting them
                specifically in this phase.
            '''),
    #html.Div(children = html.Img(src = 'assets/prefAltPlan.png', style={'height':'60%', 'width':'60%'}),style={"textAlign": "center"}),
    html.H2("What does this DSS do?"),
    html.P('''
                Firstly, this DSS's foundation is what each decision-maker considers as their objective or goal to be achieved in this project. 
                More specifically, it calculates the project's NPV (Energy Provider), Noise Pollution in Oss and Den Bosch (Local Residents),
                Bird Mortality (Ecologists), Particle Pollution (RIVM), Energy Yield for each municipality and Project Time (Municipalities).
                The formulas used to calculate these parameters are taken from academic literature or empiric estimates from the construction industry,
                ensuring that any solution within the given boundaries is realistic and realizable.
                '''),
])
