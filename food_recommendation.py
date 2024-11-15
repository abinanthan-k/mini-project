import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import json

# Load recipe data
recipe_df = pd.read_csv("recipe_final (1).csv")
recipe_df["ingredients_list"] = recipe_df["ingredients_list"].fillna('')
scaler = StandardScaler()
vectorizer = TfidfVectorizer()

X_numerical = scaler.fit_transform(recipe_df[['calories', 'protein']])
X_ingredients = vectorizer.fit_transform(recipe_df['ingredients_list'])
X_combined = np.hstack([X_numerical, X_ingredients.toarray()])

app = Flask(__name__)
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(X_combined)

def calculate_needs(height, weight, gender, age, activity_level):
    if gender.lower() == 'male':
        bmr = 66.47 + (13.7 * weight) + (5 * height) - (6.8 * age)
    elif gender.lower() == 'female':
        bmr = 655.1 + (9.6 * weight) + (1.8 * height) - (4.7 * age)
    else:
        raise ValueError("Gender should be either 'male' or 'female'")
    
    activity_factors = {
        'sedentary': 1.2,
        'lightly active': 1.375,
        'moderately active': 1.55,
        'very active': 1.725,
        'extra active': 1.9
    }
    caloric_needs = bmr * activity_factors.get(activity_level, 1.2)
    protein_needs = weight
    return caloric_needs, protein_needs

def recommend(height, weight, gender, age, activity_level):
    caloric_needs, protein_needs = calculate_needs(height, weight, gender, age, activity_level)
    meal_limits = {
        'breakfast': (caloric_needs * 0.25, protein_needs * 0.25),
        'lunch': (caloric_needs * 0.35, protein_needs * 0.35),
        'dinner': (caloric_needs * 0.30, protein_needs * 0.30),
        'snack': (caloric_needs * 0.10, protein_needs * 0.10)
    }
    
    recommendations = {}
    for meal_type, (calorie_limit, protein_limit) in meal_limits.items():
        filtered_recipes = recipe_df[(recipe_df['calories'] <= calorie_limit) & (recipe_df['protein'] <= protein_limit)]
        
        meal_recommendations = filtered_recipes.sample(n=3) if not filtered_recipes.empty else []
        
        recommendations[meal_type] = meal_recommendations[['recipe_name', 'ingredients_list', 'image_url']].to_dict('records')
    
    return recommendations

def calculate_bmi(height, weight):
    return int(((weight)/ (height*height)) * 10000)

@app.route('/')
def index():
    """Render the main input form."""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Handle form submission and return recommendations."""
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    age = int(request.form['age'])
    gender = request.form['gender']
    activity_level = request.form['activity_level']   
 
    recommendations = recommend(height, weight, gender, age, activity_level)
    bmi_value = calculate_bmi(height, weight)
    
    # Pass recommendations and BMI value to result page
    return redirect(url_for('result', recommendations=json.dumps(recommendations), bmi=bmi_value))

@app.route("/result")
def result():
    """Display the recommendations and BMI."""
    recommendations_json = request.args.get('recommendations')
    bmi_value = request.args.get('bmi')

    recommendations_dict = json.loads(recommendations_json) if recommendations_json else {}
    
    return render_template('result.html', recommendations=recommendations_dict, bmi=bmi_value)

if __name__ == '__main__':
    app.run(debug=True)