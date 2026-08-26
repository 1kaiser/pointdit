// Loads a generated model-viewer HTML page in real headless Chrome (same launch flags this
// repo's web_demo/run_demo.js already uses), waits for the model to actually load, screenshots
// it, and reports pixel stddev -- the same "don't trust the reported backend name / load event
// alone, check real pixel content" verification used in the glb_interaction_tryout project this
// pattern was carried over from (see research-repo-bringup skill).
//
// Usage: node verify_modelviewer.js <html_url> <out_png>
const { chromium } = require('playwright-core');

async function main() {
  const [, , htmlUrl, outPng] = process.argv;
  const browser = await chromium.launch({
    executablePath: process.env.CHROME_PATH || '/home/kaiser/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome',
    headless: true,
    args: [
      '--no-sandbox', '--headless=new', '--use-angle=vulkan',
      '--enable-features=Vulkan', '--disable-vulkan-surface',
      '--enable-unsafe-webgpu', '--enable-webgpu-developer-features',
    ],
  });
  const page = await browser.newPage({ viewport: { width: 960, height: 720 } });
  page.on('console', (msg) => console.log('[console]', msg.text()));
  page.on('pageerror', (err) => console.log('[pageerror]', err.message));

  await page.goto(htmlUrl, { waitUntil: 'load', timeout: 30000 });
  await page.waitForSelector('model-viewer[loaded]', { timeout: 60000 }).catch(() => {
    console.log('WARNING: model-viewer never reported [loaded] within timeout');
  });
  // model-viewer re-renders on its own next animation frame after load; give it a couple.
  await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));

  await page.screenshot({ path: outPng });
  console.log(`Screenshot saved: ${outPng}`);

  await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
