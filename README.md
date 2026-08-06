# Predicción USDT/VES en la nube — sin servidor, sin PC encendida

Todo el pipeline vive en **GitHub Actions**: la recolección, el
entrenamiento y la inferencia corren en los servidores de GitHub cada
pocos minutos, gratis. El propio repositorio es la base de datos
(patrón "git scraping").

```
[collect.yml]  cada ~5-10 min en GitHub Actions
   ├── collector.py → data/AAAA-MM-DD.csv  +  data/books/*.jsonl
   ├── predict.py   → prediction.json      (si ya existe model.pkl)
   └── git commit + push  (el repo ES la base de datos)

[train.yml]  domingos + manual
   └── train.py → model.pkl  (LightGBM, walk-forward validation)

[Tu dashboard en Netlify]
   └── fetch('https://raw.githubusercontent.com/TU_USUARIO/TU_REPO/main/prediction.json')
```

## Puesta en marcha (10 minutos, una sola vez)

1. Crea un repositorio **público** en GitHub (ej. `p2p-ves-predictor`).
   Tiene que ser público: con repos públicos los minutos de Actions son
   ilimitados y gratis; con privados el cupo mensual no alcanza para
   288 corridas diarias. Los datos son públicos de todas formas (precios
   de mercado).

2. Sube todos los archivos de esta carpeta (incluyendo `.github/`):

   ```
   git init
   git add .
   git commit -m "pipeline inicial"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/p2p-ves-predictor.git
   git push -u origin main
   ```

3. En GitHub → pestaña **Actions** → habilita los workflows si lo pide.
   Luego **Settings → Actions → General → Workflow permissions** →
   marca **"Read and write permissions"** (necesario para que el bot
   haga commit de los datos).

4. Prueba manual: Actions → workflow **collect** → **Run workflow**.
   Si en 1-2 minutos aparece `data/2026-08-XX.csv` en el repo, ya está
   vivo. A partir de ahí corre solo, para siempre, con tu PC apagada.

5. En 2–4 semanas: Actions → workflow **train** → **Run workflow**.
   Revisa el log: si la accuracy walk-forward supera al baseline, el
   modelo empieza a publicar `prediction.json` en cada snapshot.

## Conectar tu dashboard (Netlify)

```js
const URL = 'https://raw.githubusercontent.com/TU_USUARIO/p2p-ves-predictor/main/prediction.json';
const pred = await (await fetch(URL, {cache: 'no-store'})).json();
// pred.signal → "SUBE" | "LATERAL" | "BAJA"
// pred.probs  → {baja, lateral, sube}
// pred.confidence, pred.brecha_pct, pred.rsi_14, pred.imbalance ...
```

`raw.githubusercontent.com` permite CORS, así que funciona directo desde
tu sitio estático sin proxy ni backend.

## Cosas que debes saber (limitaciones reales)

- **El cron de Actions no es puntual.** "Cada 5 min" en la práctica es
  cada 5–12 min, con retrasos en horas pico de GitHub. El pipeline ya lo
  tolera: los features se calculan sobre una rejilla re-muestreada y las
  etiquetas se asignan por tiempo real, no por número de filas.
- **Posible bloqueo de IP.** Binance a veces filtra tráfico de
  datacenters. Si los logs muestran errores 403/451 constantes, el plan B
  es mover solo `collector.py` a un VPS gratuito (Oracle Cloud Always
  Free) y dejar el resto igual. Pruébalo primero: en muchos casos
  funciona sin problemas.
- **Actividad del repo.** GitHub desactiva crons en repos sin actividad
  por 60 días — irrelevante aquí, porque cada snapshot es un commit.
- **Crecimiento del repo.** Los libros completos (`data/books/`) pesan
  ~1 MB/día. Al cabo de meses puedes archivar los JSONL viejos en un
  release y borrarlos del árbol si molesta.

## Los archivos

- `collector.py` — snapshot de Binance P2P (libro de 20×2 anuncios) + BCV
- `features.py` — RSI, EMA ratio, Bollinger squeeze, imbalance y su
  velocidad, brecha BCV y su aceleración, quincena, horario bancario
- `train.py` — LightGBM 3 clases (sube/lateral/baja) + walk-forward
- `predict.py` — inferencia del último snapshot → `prediction.json`

## Nota honesta

Ningún modelo predice este mercado con certeza, y el P2P venezolano
sufre shocks exógenos (anuncios oficiales, cortes, pánicos) que ningún
feature captura a tiempo. El objetivo realista es superar al azar de
forma consistente y medible — trátalo como probabilidad informativa,
no como orden de operación.
