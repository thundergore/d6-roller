# 🎲 D6 Roller & Combat Resolver

A powerful dice roller and wargaming combat resolution tool with advanced statistical analysis. Perfect for Warhammer, Age of Sigmar, and any d6-based tabletop games.

![Python](https://img.shields.io/badge/python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/streamlit-web%20app-red)

## ✨ Features

### 🎲 Simple Dice Roller
- Roll 1-1000 d6 dice with detailed statistics
- Visual distribution charts
- Historic roll tracking
- Quick roll buttons (5, 10, 20 dice)

### ⚔️ Combat Resolver
Complete wargaming combat resolution with:

**Combat Phases:**
- To Hit → To Wound → Saves → Ward Saves

**Advanced Mechanics:**
- ✓ Reroll 1s and reroll all fails
- ✓ +/- modifiers for all phases
- ✓ Damage characteristics (1-6) with modifiers
- ✓ 6s auto-wound (skip wound phase)
- ✓ 6s deal mortal wounds (skip wound + save)

**📊 Statistical Analysis:**
- Expected damage using probability theory
- Actual vs expected comparison
- Convergence tracking over multiple combats
- Luck indicators (running hot/cold/balanced)
- Standard deviation and variance
- **Watch the Law of Large Numbers in action!**

## 🚀 Quick Start

### Option 1: Run Web App Locally (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Opens automatically at `http://localhost:8501` ✨

### Option 2: Deploy Online for FREE

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "D6 Roller app"
   git remote add origin https://github.com/yourusername/d6-roller.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app" → Select your repo → Deploy!

3. **Share your link!**
   - Your app will be live at: `https://yourusername-d6-roller.streamlit.app`
   - Anyone can use it - no installation needed!
   - Works on mobile! 📱

### Option 3: Desktop App (Traditional)

```bash
pip install numpy
python d6roller_app.py
```

Runs as a traditional Tkinter desktop application.

## 📖 How to Use

### Simple Dice Rolling

1. Click the "🎲 Simple Rolls" tab
2. Enter number of dice (or use quick buttons)
3. Click "Roll Dice"
4. View distribution, statistics, and historic tracking

### Combat Resolution

1. Click the "⚔️ Combat Resolver" tab
2. **Configure your attack:**
   - Number of attacks
   - Damage characteristic
   - Hit target (e.g., 3+) with modifiers
   - Optional: Rerolls, auto-wounds, mortal wounds
3. **Configure wound phase:**
   - Wound target with modifiers
   - Optional: Rerolls
4. **Configure defender:**
   - Save value with modifiers
   - Optional: Ward save (5+, 6+, etc.)
5. Click "Resolve Combat"

### Understanding the Statistics

**Expected Damage:** What probability theory predicts you'll roll

**Actual Damage:** What you actually rolled

**Deviation:** The difference between expected and actual
- Positive = Lucky rolls! 🔥
- Negative = Unlucky rolls ❄️
- Near zero = Balanced ⚖️

**Convergence:** As you resolve more combats with the same configuration:
- Your average damage approaches the expected value
- This demonstrates the **Law of Large Numbers**!
- Run 10-20 combats to see meaningful patterns

## 🎯 Example Scenarios

### Example 1: Basic Infantry Attack
```
10 attacks, Damage 1
Hit on 3+, Wound on 4+
Enemy: Save 4+, Ward 5+
Expected: ~2.3 damage
```
Resolve multiple times to see convergence!

### Example 2: Elite Strike with Mortals
```
20 attacks, Damage 2
Hit on 2+ with "6s = mortal wounds"
Wound on 3+
Enemy: Save 3+, Ward 6+
```
Watch natural 6s bypass saves completely!

### Example 3: High-Strength Weapon
```
15 attacks, Damage 3 (+1 damage modifier)
Hit on 3+ with "6s auto-wound"
Wound on 5+ (but 6s skip this!)
Enemy: Save 2+, Ward 5+
```
Auto-wounds help against tough targets!

## 📊 The Math Behind It

Expected damage is calculated using probability theory:

**Basic Formula:**
```
Expected = Attacks × P(hit) × P(wound) × P(fail_save) × P(fail_ward) × Damage
```

**Reroll Mechanics:**
- Reroll 1s: `P + (1/6 × P)`
- Reroll all fails: `P + ((1-P) × P)`

**Special Rules:**
- Mortal wounds skip wound + save phases
- Auto-wounds skip wound phase only
- Each calculated separately, then summed

The statistical analysis shows whether you're having a "lucky" or "unlucky" game!

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit** - Modern web framework
- **NumPy** - Fast dice rolling & statistics
- **Tkinter** - Optional desktop version

## 📁 Project Structure

```
d6-roller/
├── app.py                    # Streamlit web app (USE THIS!)
├── d6roller_app.py          # Tkinter desktop app (alternative)
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── README.md                # This file
```

## 🌟 Why Streamlit?

✅ **Easy to share** - Send a link, no downloads needed
✅ **Works everywhere** - Desktop, mobile, tablet
✅ **Free hosting** - Streamlit Cloud is completely free
✅ **Auto-updates** - Push to GitHub, app updates automatically
✅ **Modern UI** - Clean, responsive web interface
✅ **No building** - No executables to compile

## 💡 Tips & Tricks

1. **Understanding Variance**
   - Single combats vary wildly from expected
   - Run 10+ combats for meaningful patterns
   - 20+ combats show clear convergence

2. **Testing Strategies**
   - Use expected damage to compare weapon profiles
   - See how much rerolls actually help
   - Evaluate special rules objectively

3. **Clear Statistics**
   - Use "Clear Statistics" when switching matchups
   - Each configuration should be tested independently

4. **Mobile Usage**
   - Web app works great on phones/tablets
   - Perfect for quick calculations during games

## 🤝 Contributing

Want to add features? Found a bug?

1. Fork the repository
2. Make your changes to `app.py`
3. Test locally: `streamlit run app.py`
4. Submit a pull request!

**Ideas for contributions:**
- Additional dice types (d4, d8, d10, d12, d20)
- Save/load combat configurations
- Export results to PDF/CSV
- Charts and graphs for damage distribution
- Dark mode toggle
- Custom army profiles

## 📝 License

MIT License - Feel free to use, modify, and share!

## 🙏 Acknowledgments

Built for tabletop wargamers who love probability as much as rolling dice!

Perfect for Warhammer 40K, Age of Sigmar, Kill Team, and any d6-based wargaming system.

## 📞 Support

Having issues?
- Check that Python 3.8+ is installed
- Ensure `requirements.txt` dependencies are installed
- Try clearing browser cache for web version
- Check console for error messages

## 🎲 Have Fun!

May the odds be ever in your favor!

---

**Made with ❤️ for the tabletop gaming community**

[Demo Link] | [Report Bug] | [Request Feature]
