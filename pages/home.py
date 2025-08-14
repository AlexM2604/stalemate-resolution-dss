from dash import html

home_layout = html.Div([
    html.H1("Decision Support System (DSS) For Resolving Stalemates"),

    html.P("This is the home page of the decision support system to resolve decision-making stalemates in complex construction projects. Here you can read about project background and the main functions of this tool. As well as that, extra project information is accessible via the following links:"),

    html.Div([html.A(
        'Formula Sheet',
        href='/assets/Formula_Sheet_21_04_25.pdf',  # Path relative to assets folder
        download='yourfile.pdf',      # Suggested filename for download
        target='_blank'              # Open in new tab (optional)
    )]),
    html.Div([html.A(
        'User Manual',
        href='/assets/User_Manual.pdf',  # Path relative to assets folder
        download='yourfile.pdf',      # Suggested filename for download
        target='_blank'              # Open in new tab (optional)
    )]),
    html.Div([html.A(
        'Stalemate Resolution Mechanism',
        href='/assets/Stalemate_Resolution_Mechanism.pdf',  # Path relative to assets folder
        download='yourfile.pdf',      # Suggested filename for download
        target='_blank'              # Open in new tab (optional)
    )]),

    html.H2("Duurzame Polder - Sustainable Energy for cities of Den Bosch and Oss"),
    html.P('''
                Duurzame Polder is an ongoing sustainable energy project initiated by the municipalities of Den Bosch and Oss,
                which are both part of the Noord-Brabant Province of the Netherlands. It started from a province-wide
                search for a location to put a large sustainable energy project to achieve sustainable energy generation
                goals. It was decided that the most fitting option is the polder located between den Bosch and Oss, with
                municipality borders going through it. The project has been going through various exploration and assessment phases since 2017
                and is now in the stage of collecting feedback for one of the design variants. Here is the project website (in Dutch):
                https://www.duurzamepolder.nl/. During the latest stage of the project several alternatives were analyzed and the
                "Preffered Alternative" was chosen as the one to be used in acquiring the permits.
                '''),
    html.Div(children = [html.Img(src = 'assets/polderPlan.png', style={'height':'40%', 'width':'40%'}),
                         html.Img(src = 'assets/prefAltPlan.png', style={'height':'40%', 'width':'40%'})],style={"textAlign": "center"}),
    html.P('''
                The "Preferred Alternative" under consideration makes nobody happy.
                It performs badly on "noise pollution" and "effect on bird habitats" parameters, while barely 
                accomplishing the energy goals of each municipality. The complexity of this project stems from the conflicting interests of the involved
                stakeholders. In this case, the most apparent conflict is between residents (noise pollution) and ecologists (bird habitats) on one side
                and the municipalities on the other. In the media landscape, accusations of using outdated wind turbine modeling have been brought up,
                along with concern for additional health effects of the project. This led to a pause in the project development process to collect 
                more feedback from all stakeholders and make changes to the Preffered Alternative.
            '''),
    html.P('''
                In order to see if a group optimal solution exists for the Duurzame Polder project, Preferendus modelling was applied to this case
                by Teuber & Wolfert (2024). This application demonstrated that a decision-making stalemate exists in the Duurzame Polder project.
                For this reason, the application of Preferendus to this project will be extended, along with the development of the stalemate
                resolution mechanism. This is done in completion of the Master Thesis for the programme of Construction, Management and Engineering
                at TU Delft by Alexey Matyunin.
            '''),
    #html.Div(children = html.Img(src = 'assets/prefAltPlan.png', style={'height':'60%', 'width':'60%'}),style={"textAlign": "center"}),
    html.H2("What does this DSS do?"),
    html.P('''
                This DSS allows decision-makers in the Duurzame Polder project to interact with the mathematical model of the project (Preferendus), which can generate
                group optimal design variants. For more information on the mathematical aspect of the model, please check the
                Formula Sheet linked at the top of the page. To generate the group optimal design variant, the model requires decision-makers of the
                project to enter their project goals in the Preference & Weights tab. The goals of all decision-makers in the project can be 
                viewed via the Overview tab. After all goals have been defined, the group optimal design variant can be generated in the Dashboard tab.
                
                '''),
    html.H2("How does this DSS resolve stalemates?"),
    html.P('''
                 This model can help to resolve stalemates in three capacities:
                '''),
    html.Li('Identification'),
    html.Li('Consultation'),
    html.Li('Mediation'),
    html.P('''
            Firstly, application of preference function modelling and Preferendus allows to easily identify a situation with an existing stalemate. In this DSS
            a stalemate is identified by the unacceptability of the group optimal solution. For more information please see the User Guide document.
                '''),
    html.P('''
            The consultation method means that the model enables decision-makers to find the solution for the stalemate by themselves. It can act as a consultant in
            the negotiation process and dynamically change the group optimal solution according to the changing goals and preferences of the decision-makers.
                '''),
    html.P('''
            The mediation approach features the use of the Stalemate Resolution Mechanism. In it, the model recommends the minimal extension of acceptable project
            outcomes to one or several decision-makers that will result in a group optimal solution acceptable to all parties. For more details on how the Stalemate
            Resolution Mechanism works please check one of the links at the top of this page. Due to the high computation time requirements of the mechanism, it is
            not included into the interactive version of the model. Instead it is applied in advance of the decision-making sessions after initial preferences were
            defined. The generated preference recommendation is then presented at the decision-making session (currently available in the Stalemate Resolution tab).
                ''')
],style={'marginLeft': 100, 'marginRight': 100})
