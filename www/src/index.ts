import init, { init_renderer } from '../pkg/render.js';

async function main() {
    await init();
    init_renderer();
}

main();
