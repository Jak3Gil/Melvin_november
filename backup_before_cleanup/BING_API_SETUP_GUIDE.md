# 🔍 Bing API Setup Guide for Melvin

## Step 1: Get Your Free Bing API Key

### 1.1 Create Microsoft Azure Account
- Go to: https://portal.azure.com/
- Sign in with your Microsoft account (or create one)
- **Free tier available** - no credit card required for basic usage

### 1.2 Create Bing Search Resource
1. In Azure portal, click **"Create a resource"**
2. Search for **"Bing Search v7"**
3. Click **"Create"**
4. Fill in the details:
   - **Subscription**: Choose your Azure subscription
   - **Resource Group**: Create new or use existing
   - **Resource Name**: `melvin-bing-search` (or any unique name)
   - **Pricing Tier**: **F1 (Free)** - 1,000 queries/month
   - **Region**: Choose closest to you
5. Click **"Review + Create"** → **"Create"**

### 1.3 Get Your API Key
1. Wait for deployment to complete (2-3 minutes)
2. Go to your new Bing Search resource
3. Click **"Keys and Endpoint"** in left menu
4. Copy **Key 1** (keep it secure!)

## Step 2: Configure Melvin

### 2.1 Set Environment Variable (Temporary)
```bash
set BING_API_KEY=your_actual_api_key_here
```

### 2.2 Set Environment Variable (Permanent)
1. Press `Win + R`, type `sysdm.cpl`, press Enter
2. Click **"Environment Variables"**
3. Under **"User variables"**, click **"New"**
4. Variable name: `BING_API_KEY`
5. Variable value: `your_actual_api_key_here`
6. Click **"OK"** on all dialogs

### 2.3 Test Melvin
```bash
melvin_bing.exe
```

## Step 3: Verify It's Working

Ask Melvin: **"What is the latest news about AI?"**

If working correctly, you'll see:
- "🔍 Bing API key loaded - web search enabled!"
- Real web search results in responses
- "I looked this up for you: [actual web content]"

## Free Tier Limits
- **1,000 queries per month**
- **Perfect for personal use**
- **No credit card required**

## Troubleshooting

### "BING_API_KEY not set"
- Make sure you set the environment variable
- Restart command prompt after setting

### "Web search failed"
- Check your API key is correct
- Verify you're within monthly limits
- Check internet connection

### "No search results found"
- API key might be invalid
- Check Azure portal for any errors

## Security Notes
- **Never share your API key**
- **Don't commit it to public repositories**
- **Use environment variables for security**

---

**Ready to give Melvin real web search power!** 🧠✨
