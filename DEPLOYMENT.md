# Heroku Deployment Guide

## Prerequisites
1. A [GitHub](https://github.com) account
2. A [Heroku](https://heroku.com) account (free tier available)
3. Git installed on your computer
4. The Heroku CLI (optional, but recommended)

## Step 1: Push Your Code to GitHub

### Initialize Git Repository (if not already done)
```powershell
cd "c:\Users\user\Documents\Covenant University\300Lvl Alpha\CSC334\Assignments\ENEASATO_23CG034068_EMOTION_DETECTION_WEB_APP"
git init
```

### Add All Files
```powershell
git add .
```

### Commit Your Changes
```powershell
git commit -m "Initial commit - Emotion Detection Web App"
```

### Create a New Repository on GitHub
1. Go to https://github.com/new
2. Name your repository (e.g., `emotion-detection-app`)
3. **DO NOT** initialize with README, .gitignore, or license (you already have these)
4. Click "Create repository"

### Push to GitHub
```powershell
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
git branch -M main
git push -u origin main
```

Replace `YOUR-USERNAME` and `YOUR-REPO-NAME` with your actual GitHub username and repository name.

## Step 2: Deploy to Heroku

### Option A: Deploy via Heroku Dashboard (Recommended for Beginners)

1. **Log in to Heroku**
   - Go to https://dashboard.heroku.com/

2. **Create New App**
   - Click "New" → "Create new app"
   - Choose a unique app name (e.g., `emotion-detector-app-yourname`)
   - Select a region (United States or Europe)
   - Click "Create app"

3. **Connect to GitHub**
   - In the "Deploy" tab, select "GitHub" as deployment method
   - Click "Connect to GitHub"
   - Search for your repository name
   - Click "Connect"

4. **Enable Automatic Deploys (Optional)**
   - Scroll to "Automatic deploys" section
   - Click "Enable Automatic Deploys"
   - This will deploy automatically whenever you push to GitHub

5. **Manual Deploy**
   - Scroll to "Manual deploy" section
   - Select the `main` branch
   - Click "Deploy Branch"

6. **Wait for Build**
   - Heroku will install dependencies and start your app
   - This may take 5-10 minutes

7. **View Your App**
   - Click "View" or "Open app" to see your live application
   - Copy the URL (e.g., `https://your-app-name.herokuapp.com`)

### Option B: Deploy via Heroku CLI

1. **Install Heroku CLI**
   - Download from: https://devcenter.heroku.com/articles/heroku-cli

2. **Login to Heroku**
   ```powershell
   heroku login
   ```

3. **Create Heroku App**
   ```powershell
   cd "c:\Users\user\Documents\Covenant University\300Lvl Alpha\CSC334\Assignments\ENEASATO_23CG034068_EMOTION_DETECTION_WEB_APP"
   heroku create your-app-name
   ```

4. **Deploy**
   ```powershell
   git push heroku main
   ```

5. **Open Your App**
   ```powershell
   heroku open
   ```

## Step 3: Important Notes

### File Size Limitations
⚠️ **WARNING**: Your `emotion_guardian_model.h5` file might be too large for GitHub (>100MB).

**Solutions:**

1. **Use Git Large File Storage (Git LFS)**
   ```powershell
   git lfs install
   git lfs track "*.h5"
   git add .gitattributes
   git commit -m "Add Git LFS tracking for model file"
   git push
   ```

2. **Store Model Externally** (Recommended for large files)
   - Upload to Google Drive, Dropbox, or AWS S3
   - Modify `app.py` to download the model on startup
   - Add download logic before loading the model

3. **Use Heroku Slug Size Limit**
   - Heroku has a 500MB slug size limit
   - If your model is <100MB, it should work fine

### Database Persistence
⚠️ SQLite databases on Heroku are **ephemeral** (reset on each deployment).

**For Production:**
- Use Heroku Postgres addon:
  ```powershell
  heroku addons:create heroku-postgresql:mini
  ```
- Update your code to use PostgreSQL instead of SQLite

### Environment Variables
If you need to set environment variables:
```powershell
heroku config:set DEBUG=False
```

## Step 4: Verify Deployment

1. Open your app URL
2. Test image upload functionality
3. Test webcam capture (may require HTTPS)
4. Check the logs if something doesn't work:
   ```powershell
   heroku logs --tail
   ```

## Troubleshooting

### Build Fails
- Check `heroku logs --tail` for errors
- Verify all dependencies in `requirements.txt`
- Ensure `Procfile` is correctly named (no extension)

### App Crashes
- Check logs: `heroku logs --tail`
- Verify your model file exists
- Check file paths are correct

### Webcam Not Working
- HTTPS is required for webcam access
- Heroku apps get HTTPS by default

### Slow Performance
- Free Heroku dynos sleep after 30 minutes of inactivity
- Consider upgrading to hobby or professional dynos

## Files Created for Heroku

✅ **Procfile** - Tells Heroku how to run your app
✅ **runtime.txt** - Specifies Python 3.11.6
✅ **.gitignore** - Prevents unnecessary files from being committed
✅ **static/uploads/.gitkeep** - Ensures uploads directory exists
✅ **app.py** - Updated to use PORT environment variable

## Updating Your App

When you make changes:
```powershell
git add .
git commit -m "Description of changes"
git push origin main
```

If you enabled automatic deploys, Heroku will automatically deploy the changes.
Otherwise, manually deploy from the Heroku dashboard.

## Save Your App URL

Once deployed, save your Heroku app URL to `link_to_my_web_app.txt`:
```
https://your-app-name.herokuapp.com
```

---

**Need Help?**
- Heroku Documentation: https://devcenter.heroku.com/
- Heroku Support: https://help.heroku.com/
