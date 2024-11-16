import os
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# Ustaw ścieżkę bazową
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_DIR = os.path.join(BASE_DIR, "users")

# Upewnij się, że katalog `users` istnieje
os.makedirs(USERS_DIR, exist_ok=True)

# Rejestracja użytkownika
@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username')
    password = request.form.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    if not username.isalnum():
        return jsonify({"error": "Username can only contain letters and numbers"}), 400

    # Tworzenie folderu dla użytkownika
    user_dir = os.path.join(USERS_DIR, username)
    if os.path.exists(user_dir):
        return jsonify({"error": "User already exists"}), 400

    os.makedirs(user_dir)
    user_file = os.path.join(user_dir, "user.txt")

    # Zapis danych użytkownika
    with open(user_file, 'w') as f:
        f.write(f"Username: {username}\nPassword: {password}")

    return jsonify({"message": "User registered successfully"}), 200

# Logowanie użytkownika
@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    user_dir = os.path.join(USERS_DIR, username)
    user_file = os.path.join(user_dir, "user.txt")

    if not os.path.exists(user_file):
        return jsonify({"error": "User does not exist"}), 404

    # Odczyt danych użytkownika
    with open(user_file, 'r') as f:
        stored_password = f.readlines()[1].split(":")[1].strip()

    if stored_password != password:
        return jsonify({"error": "Invalid password"}), 400

    return jsonify({"message": "Login successful"}), 200

# Serwowanie plików statycznych (HTML, CSS, JS)
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(BASE_DIR, filename)

# Uruchom serwer
if __name__ == '__main__':
    app.run(debug=True)
