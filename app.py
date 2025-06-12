from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Chargement des paramètres du modèle deep learning
with open('params_model.pkl', 'rb') as f:
    params = pickle.load(f)

with open('normalization_stats.pkl', 'rb') as f:
    normalization_stats = pickle.load(f)

mu = normalization_stats['mu']
sigma = normalization_stats['sigma']


with open('categories.pkl', 'rb') as f:
    cat_cols = pickle.load(f)
# Colonnes catégorielles à encoder et leurs options
category_options = {
    'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
    'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 
                 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 
                 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 
                 'Preschool'],
    'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 
                      'Separated', 'Widowed', 'Married-spouse-absent', 
                      'Married-AF-spouse'],
    'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
                  'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
                  'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 
                  'Transport-moving', 'Priv-house-serv', 'Protective-serv', 
                  'Armed-Forces'],
    'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 
                    'Other-relative', 'Unmarried'],
    'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Black', 'Other'],
    'sex': ['Male', 'Female'],
    'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 
                      'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 
                      'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 
                      'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 
                      'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 
                      'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 
                      'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 
                      'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 
                      'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
}

# Liste des features dans l'ordre de ton modèle 
features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
            'marital-status', 'occupation', 'relationship', 'race', 'sex', 
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

# Fonctions du modèle deep learning 
def relu(Z): return np.maximum(0, Z)
def sigmoid(Z): return 1 / (1 + np.exp(-Z))

def forward_propagation(X, parametres):
    activations = {'A0': X}
    C = len(parametres) // 2
    for c in range(1, C):
        Z = parametres['W' + str(c)] @ activations['A' + str(c-1)] + parametres['b' + str(c)]
        A = relu(Z)
        activations['Z' + str(c)] = Z
        activations['A' + str(c)] = A
    ZC = parametres['W' + str(C)] @ activations['A' + str(C-1)] + parametres['b' + str(C)]
    AC = sigmoid(ZC)
    activations['Z' + str(C)] = ZC
    activations['A' + str(C)] = AC
    return activations

def predict(X, parametres):
    A = forward_propagation(X, parametres)['A' + str(len(parametres)//2)]
    return (A >= 0.5).astype(int).flatten()[0]

def preprocess_input(form_data):
    x = []
    for feat in features:
        val = form_data.get(feat, '')
        
        # Traitement spécial pour les champs catégoriels
        if feat in category_options:
            # Trouver l'index de la valeur sélectionnée
            options = category_options[feat]
            try:
                val = options.index(val) if val in options else 0
            except:
                val = 0
        else:
            # Pour les champs numériques
            try:
                val = float(val) if val else 0.0
            except:
                val = 0.0
        
        x.append(val)
    x = np.array(x)
    # Normalisation
    x_norm = (x - mu) / sigma
    return x_norm.reshape(-1,1)  # Colonne (n_features, 1) pour la matrice de poids

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if request.is_json:
                data = request.get_json()
                X_input = preprocess_input(data)
                pred = predict(X_input, params)
                resultat = "Income is greater than 50K" if pred == 1 else "Income is less than or equal to 50K"
                return jsonify({"resultat": resultat})
            else:
                return jsonify({"resultat": "Requête non JSON"}), 400
        except Exception as e:
            return jsonify({"resultat": f"Erreur lors de la prédiction : {str(e)}"}), 500

    return render_template("index.html", category_options=category_options, features=features)

@app.route('/apropos')
def apropos():
    return render_template('apropos.html')

if __name__ == "__main__":
    app.run(debug=True)