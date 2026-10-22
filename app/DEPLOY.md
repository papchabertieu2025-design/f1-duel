# Put the website online with PythonAnywhere (about 10 minutes)

1. **Upload the app.** On pythonanywhere.com, open *Files* and create the folder `f1duel_app`.
   Upload `app.py`, `model.json`, and the folder `templates/` (with `index.html`).
   Or open a *Bash console* and run `git clone https://github.com/<you>/f1-duel.git`.
2. **Install the libraries.** In a *Bash console*: `pip install --user flask openai`
3. **Create the web app.** *Web* tab → *Add a new web app* → *Flask* → Python 3.x →
   path to `app.py` (for example `/home/<username>/f1-duel/app/app.py`).
4. **Add the OpenAI key (AI box).** *Web* tab → open the *WSGI configuration file* and add, above the import:
   ```python
   import os
   os.environ["OPENAI_API_KEY"] = "sk-..."      # key from platform.openai.com > API keys
   os.environ["OPENAI_MODEL"] = "gpt-4o-mini"   # cheap and fast; change if your course uses another
   ```
   Never put the key in GitHub. Without a key the site still works; only the AI box shows a message.
5. **Reload** (green button on the *Web* tab) and open `https://<username>.pythonanywhere.com`.

**Update before each race:** run `python -m f1duel.forecast 2026 <round>` on your computer, upload the new
`model.json` (or `git pull` in the PythonAnywhere console), and press *Reload*.
