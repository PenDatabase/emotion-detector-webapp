# 🚀 Quick Deployment Guide

## Prerequisites Checklist
- [ ] GitHub account created
- [ ] Heroku account created
- [ ] Git installed on your computer

## Step 1: Push to GitHub (Choose One Method)

### Method A: Use the Automated Script (Recommended)
```powershell
.\setup-github.ps1
```
Follow the prompts to set up and push to GitHub.

### Method B: Manual Setup
```powershell
# Initialize Git
git init

# Add files
git add .

# Commit
git commit -m "Initial commit - Emotion Detection Web App"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy to Heroku

### Via Heroku Dashboard (Easiest)
1. Go to https://dashboard.heroku.com/
2. Click "New" → "Create new app"
3. Choose an app name (e.g., `emotion-detector-yourname`)
4. Click "Create app"
5. Go to "Deploy" tab
6. Select "GitHub" as deployment method
7. Connect your repository
8. Click "Deploy Branch"
9. Wait for build to complete
10. Click "View" to see your app!

### Via Heroku CLI (Alternative)
```powershell
# Install Heroku CLI first from: https://devcenter.heroku.com/articles/heroku-cli

heroku login
heroku create your-app-name
git push heroku main
heroku open
```

## Step 3: Save Your URL
Once deployed, save your app URL to `link_to_my_web_app.txt`:
```
https://your-app-name.herokuapp.com
```

## Files Created for Deployment
✅ `Procfile` - Tells Heroku how to run your app  
✅ `runtime.txt` - Specifies Python version  
✅ `.gitignore` - Prevents unnecessary files from being committed  
✅ `setup-github.ps1` - Automated Git setup script  
✅ `DEPLOYMENT.md` - Detailed deployment instructions  

## Need More Help?
See `DEPLOYMENT.md` for detailed instructions and troubleshooting.

## Troubleshooting
- **Build fails**: Check `requirements.txt` and ensure all dependencies are correct
- **App crashes**: Run `heroku logs --tail` to see error messages
- **Model not loading**: Verify `emotion_guardian_model.h5` is committed to Git

---

Good luck with your deployment! 🎉
