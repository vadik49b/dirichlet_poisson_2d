'use strict';

const Nx = 51;
const Ny = 101;
const PIXEL_SIZE = 10;
const BLINK_TIME = 100;
const FRAME_TIME = 50;
const colors = ['red', 'blue', 'green', 'black']

const initCanvas = (width, height) => {
  const canvas = document.createElement('canvas');
  canvas.id = 'graphCanvas';
  canvas.width = width * PIXEL_SIZE;
  canvas.height = height * PIXEL_SIZE;
  document.body.appendChild(canvas);
  return canvas;
};

const canvas = initCanvas(Ny, Nx);
const ctx = canvas.getContext('2d');

const blink = (threadId, i, j) => {
  ctx.fillStyle = colors[threadId];
  ctx.fillRect(j * PIXEL_SIZE, i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
  setTimeout(() => {
    ctx.fillStyle = 'white';
    ctx.fillRect(j * PIXEL_SIZE, i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
  }, BLINK_TIME)
};

const main = (content) => {
  const lines = content.split('\n');
  let i = 0;
  const thread = () => {
    const timer = setInterval(() => {
      if (i < lines.length) {
        const line = lines[i++];
        const c = line.split(' ').map(n => +n);
        blink(c[0], c[1], c[2]);
      } else {
        clearInterval(timer);
      }
    }, FRAME_TIME);
  };
  // run "threads"
  thread();
};

const input = document.getElementById('fileinput');
input.addEventListener('change', (e) => {
  const file = e.target.files[0];
	const textType = /text.*/;
	if (file.type.match(textType)) {
		const reader = new FileReader();
		reader.onload = function(e) {
      console.log('File readed');
			main(reader.result);
		}
		reader.readAsText(file);
	} else {
		alert('File not supported!');
	}
}, false);

