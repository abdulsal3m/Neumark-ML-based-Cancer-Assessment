from app import app
import os

if __name__ == '__main__':
    # Use port 5001 instead of default 5000 to avoid common conflicts
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
