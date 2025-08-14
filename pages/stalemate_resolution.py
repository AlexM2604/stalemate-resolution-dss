from dash import html

stalemate_layout = html.Div([
    html.H1("Stalemate Resolution Mechanism"),

    html.P('''
    This page of the decision support system displays the results of the Stalemate Resolution Mechanism. You can find a more in-depth explanation of how the 
    mechanism works here:
    '''),

    html.Div([html.A(
        'Stalemate Resolution Mechanism',
        href='/assets/Stalemate_Resolution_Mechanism.pdf',  # Path relative to assets folder
        download='yourfile.pdf',      # Suggested filename for download
        target='_blank'              # Open in new tab (optional)
    )]),
    html.P('''
        In short, the mechanism recommends the smallest extension of acceptability regions of various decision-makers that leads to a group optimal design variant
        that scores at least 0 or above on preference for all decision-makers. In other words, it shows what outcomes every decision-maker should be willing to accept
        if they want to continue this project within this project configuration. From this point, decision-makers can either accept the recommendation and change their
        preference curves accordingly or decide to consider a different project configuration. Considering a different project configuration means that a new model "core"
        that houses all objectives functions needs to be constructed, so this decision concludes the decision-making process with this specific decision support system.
                '''),
    html.H2("Results"),
    html.P('''
        Ideally the Stalemate Resolution Mechanism would be an interactive part of this DSS just like the other functions. However,due to the computational
        load of the mechanism at this stage of development, it is executed separately from the online DSS. Here you can see the recommended extension of acceptability (orange)
        along with the group optimal solution (IMAP) and the original curve (blue) for the curves already saved to this model (see Overview tab):
            '''),

    html.Div(children=[html.Img(src='assets/RESULT1.jpg', style={'height': '40%', 'width': '40%'}),
                       html.Img(src='assets/RESULT2.jpg', style={'height': '40%', 'width': '40%'})],
             style={"textAlign": "center"}),
html.P('''
        It is important to note that, as evident from the graphs, the output of the Stalemate Resolution Mechanism requires some interpretation. This is the reason
        for including the IMAP result. There are some cases, such as the recommendation for the Energy Provider NPV curve, where the group optimal solution lies
        fully in the positive region, therefore the preference recommendation of extending acceptability for this goal is irrelevant. Meanwhile in other cases, such as
        the Noise curve for Local Residents of Oss, the extension is correct, but it goes too far.
            '''),
html.P('''
        After performing this type of analysis, the interpretation of this acceptance adjustment is that the closest group optimal solution
        to the current set of preference is to concentrate the energy generation capacity on the Oss side of the project and keep a very
        small part of it in the Den Bosch side or even remove it completely. This most likely has to do with the meadow bird reserve located in the Den Bosch part
        and the threat of bird mortality versus noise nuisance to local residents. Therefore, the optimal variant is to focus efforts on the area where it is easier
        to build. However, that means that the Den Bosch municipality has to abandon their ambitions for a renewable project.
            '''),

],style={'marginLeft': 100, 'marginRight': 100})
