'use strict';

const wrapWithDiv = html => `<div>${html}</div><br/>`;

const pix = '<div class="lamp"></div>';

const Nx = 11;
const Ny = 21;

const row = wrapWithDiv(new Array(Ny + 1).join(pix));
const lamps = new Array(Nx + 1).join(row);

document.body.outerHTML = lamps;

const data = `s 1 1
s 1 2
s 1 3
s 1 4
s 1 5
s 1 6
s 9 1
s 1 7
s 4 1
s 9 2
s 2 1
s 4 2
s 9 3
s 2 2
s 4 3
s 9 4
s 2 3
s 4 4
s 9 5
s 2 4
s 4 5
s 9 6
s 2 5
s 4 6
s 9 7
s 2 6
s 4 7
s 9 4
s 2 7
s 5 1
s 9 5
s 3 1
s 5 2
s 9 6
s 3 2
s 5 3
s 9 7
s 3 3
s 5 4
s 9 8
s 3 4
s 5 5
s 9 9
s 3 5
s 5 6
s 9 10
s 3 6
s 5 7
s 9 11
s 3 7
s 6 1
s 9 12
s 4 1
s 6 2
s 9 9
s 4 2
s 6 3
s 9 10
s 4 3
s 6 4
s 9 11
s 4 4
s 6 5
s 9 12
s 4 5
s 6 6
s 9 13
s 4 6
s 6 7
s 9 14
s 4 7
s 7 1
s 9 15
s 5 1
s 7 2
s 9 16
s 5 2
s 7 3
s 9 17
s 5 3
s 7 4
s 9 14
s 5 4
s 7 5
s 9 15
s 5 5
s 7 6
s 9 16
s 5 6
s 7 7
s 9 17
s 5 7
s 8 1
s 9 18
s 6 1
s 8 2
s 9 19
s 6 2
s 8 3
s 9 19
s 6 3
s 8 4
s 6 4
s 8 5
s 6 5
s 8 6
s 6 6
s 8 7
s 6 7
s 9 1
s 7 1
s 9 2
s 7 2
s 9 3
s 7 3
s 9 4
s 7 4
s 9 5
s 7 5
s 9 6
s 7 6
s 9 7
s 7 7
s 5 1
s 1 1
s 5 2
s 1 2
s 5 3
s 1 3
s 5 4
s 1 4
s 5 5
s 1 5
s 5 6
s 1 6
s 6 1
s 2 1
s 6 2
s 2 2
s 6 3
s 2 3
s 6 4
s 2 4
s 6 5
s 2 5
s 6 6
s 2 6
s 7 1
s 3 1
s 7 2
s 3 2
s 7 3
s 3 3
s 7 4
s 3 4
s 7 5
s 3 5
s 7 6
s 3 6
s 8 1
s 4 1
s 8 2
s 4 2
s 8 3
s 4 3
s 8 4
s 4 4
s 8 5
s 4 5
s 8 6
s 4 6
s 9 1
s 5 1
s 9 2
s 5 2
s 9 3
s 5 3
s 9 4
s 5 4
s 9 5
s 5 5
s 9 6
s 5 6
s 6 1
s 6 1
s 6 2
s 6 2
s 6 3
s 6 3
s 6 4
s 6 4
s 6 5
s 6 5
s 7 1
s 6 6
s 7 2
s 1 1
s 7 3
s 1 2
s 7 4
s 1 3
s 7 5
s 1 4
s 8 1
s 1 5
s 8 2
s 2 1
s 8 3
s 2 2
s 8 4
s 2 3
s 8 5
s 2 4
s 9 1
s 2 5
s 9 2
s 3 1
s 9 3
s 3 2
s 9 4
s 3 3
s 9 5
s 3 4
s 4 4
s 3 5
s 4 5
s 4 1
s 4 6
s 4 2
s 4 7
s 4 3
s 4 8
s 4 4
s 4 9
s 4 5
s 4 10
s 5 1
s 4 11
s 5 2
s 4 12
s 5 3
s 5 4
s 5 4
s 5 5
s 5 5
s 5 6
s 1 4
s 5 7
s 1 5
s 5 8
s 1 6
s 5 9
s 1 7
s 5 10
s 1 8
s 5 11
s 1 9
s 5 12
s 1 10
s 6 4
s 1 11
s 6 5
s 1 12
s 6 6
s 2 4
s 6 7
s 2 5
s 6 8
s 2 6
s 6 9
s 2 7
s 6 10
s 2 8
s 6 11
s 2 9
s 6 12
s 2 10
s 7 4
s 2 11
s 7 5
s 2 12
s 7 6
s 3 4
s 7 7
s 3 5
s 7 8
s 3 6
s 7 9
s 3 7
s 7 10
s 3 8
s 7 11
s 3 9
s 7 12
s 3 10
s 8 4
s 3 11
s 8 5
s 3 12
s 8 6
s 4 4
s 8 7
s 4 5
s 8 8
s 4 6
s 8 9
s 4 7
s 8 10
s 4 8
s 8 11
s 4 9
s 8 12
s 4 10
s 9 4
s 4 11
s 9 5
s 4 12
s 9 6
s 5 4
s 9 7
s 5 5
s 9 8
s 5 6
s 9 9
s 5 7
s 9 10
s 5 8
s 9 11
s 5 9
s 9 12
s 5 10
s 5 5
s 5 11
s 5 6
s 5 12
s 5 7
s 6 4
s 5 8
s 6 5
s 5 9
s 6 6
s 5 10
s 6 7
s 5 11
s 6 8
s 6 5
s 6 9
s 6 6
s 6 10
s 6 7
s 6 11
s 6 8
s 6 12
s 6 9
s 7 4
s 6 10
s 7 5
s 6 11
s 7 6
s 7 5
s 7 7
s 7 6
s 7 8
s 7 7
s 7 9
s 7 8
s 7 10
s 7 9
s 7 11
s 7 10
s 7 12
s 7 11
s 1 5
s 8 5
s 1 6
s 8 6
s 1 7
s 8 7
s 1 8
s 8 8
s 1 9
s 8 9
s 1 10
s 8 10
s 1 11
s 8 11
s 2 5
s 9 5
s 2 6
s 9 6
s 2 7
s 9 7
s 2 8
s 9 8
s 2 9
s 9 9
s 2 10
s 9 10
s 2 11
s 9 11
s 3 5
s 6 6
s 3 6
s 6 7
s 3 7
s 6 8
s 3 8
s 6 9
s 3 9
s 6 10
s 3 10
s 7 6
s 3 11
s 7 7
s 4 5
s 7 8
s 4 6
s 7 9
s 4 7
s 7 10
s 4 8
s 8 6
s 4 9
s 8 7
s 4 10
s 8 8
s 4 11
s 8 9
s 5 5
s 8 10
s 5 6
s 9 6
s 5 7
s 9 7
s 5 8
s 9 8
s 5 9
s 9 9
s 5 10
s 9 10
s 5 11
s 4 9
s 6 5
s 4 10
s 6 6
s 4 11
s 6 7
s 4 12
s 6 8
s 4 13
s 6 9
s 4 14
s 6 10
s 4 15
s 6 11
s 4 16
s 1 6
s 4 17
s 1 7
s 5 9
s 1 8
s 5 10
s 1 9
s 5 11
s 1 10
s 5 12
s 2 6
s 5 13
s 2 7
s 5 14
s 2 8
s 5 15
s 2 9
s 5 16
s 2 10
s 5 17
s 3 6
s 6 9
s 3 7
s 6 10
s 3 8
s 6 11
s 3 9
s 6 12
s 3 10
s 6 13
s 4 6
s 6 14
s 4 7
s 6 15
s 4 8
s 6 16
s 4 9
s 6 17
s 4 10
s 7 9
s 5 6
s 7 10
s 5 7
s 7 11
s 5 8
s 7 12
s 5 9
s 7 13
s 5 10
s 7 14
s 1 9
s 7 15
s 1 10
s 7 16
s 1 11
s 7 17
s 1 12
s 8 9
s 1 13
s 8 10
s 1 14
s 8 11
s 1 15
s 8 12
s 1 16
s 8 13
s 1 17
s 8 14
s 2 9
s 8 15
s 2 10
s 8 16
s 2 11
s 8 17
s 2 12
s 9 9
s 2 13
s 9 10
s 2 14
s 9 11
s 2 15
s 9 12
s 2 16
s 9 13
s 2 17
s 9 14
s 3 9
s 9 15
s 3 10
s 9 16
s 3 11
s 9 17
s 3 12
s 5 10
s 3 13
s 5 11
s 3 14
s 5 12
s 3 15
s 5 13
s 3 16
s 5 14
s 3 17
s 5 15
s 4 9
s 5 16
s 4 10
s 6 10
s 4 11
s 6 11
s 4 12
s 6 12
s 4 13
s 6 13
s 4 14
s 6 14
s 4 15
s 6 15
s 4 16
s 6 16
s 4 17
s 7 10
s 5 9
s 7 11
s 5 10
s 7 12
s 5 11
s 7 13
s 5 12
s 7 14
s 5 13
s 7 15
s 5 14
s 7 16
s 5 15
s 8 10
s 5 16
s 8 11
s 5 17
s 8 12
s 6 9
s 8 13
s 6 10
s 8 14
s 6 11
s 8 15
s 6 12
s 8 16
s 6 13
s 9 10
s 6 14
s 9 11
s 6 15
s 9 12
s 6 16
s 9 13
s 6 17
s 9 14
s 7 9
s 9 15
s 7 10
s 9 16
s 7 11
s 6 11
s 7 12
s 6 12
s 7 13
s 6 13
s 7 14
s 6 14
s 7 15
s 6 15
s 7 16
s 7 11
s 7 17
s 7 12
s 1 10
s 7 13
s 1 11
s 7 14
s 1 12
s 7 15
s 1 13
s 8 11
s 1 14
s 8 12
s 1 15
s 8 13
s 1 16
s 8 14
s 2 10
s 8 15
s 2 11
s 9 11
s 2 12
s 9 12
s 2 13
s 9 13
s 2 14
s 9 14
s 2 15
s 9 15
s 2 16
s 4 14
s 3 10
s 4 15
s 3 11
s 4 16
s 3 12
s 4 17
s 3 13
s 4 18
s 3 14
s 4 19
s 3 15
s 5 14
s 3 16
s 5 15
s 4 10
s 5 16
s 4 11
s 5 17
s 4 12
s 5 18
s 4 13
s 5 19
s 4 14
s 6 14
s 4 15
s 6 15
s 4 16
s 6 16
s 5 10
s 6 17
s 5 11
s 6 18
s 5 12
s 6 19
s 5 13
s 7 14
s 5 14
s 7 15
s 5 15
s 7 16
s 5 16
s 7 17
s 6 10
s 7 18
s 6 11
s 7 19
s 6 12
s 8 14
s 6 13
s 8 15
s 6 14
s 8 16
s 6 15
s 8 17
s 6 16
s 8 18
s 1 11
s 8 19
s 1 12
s 9 14
s 1 13
s 9 15
s 1 14
s 9 16
s 1 15
s 9 17
s 2 11
s 9 18
s 2 12
s 9 19
s 2 13
s 5 15
s 2 14
s 5 16
s 2 15
s 5 17
s 3 11
s 5 18
s 3 12
s 5 19
s 3 13
s 6 15
s 3 14
s 6 16
s 3 15
s 6 17
s 4 11
s 6 18
s 4 12
s 6 19
s 4 13
s 7 15
s 4 14
s 7 16
s 4 15
s 7 17
s 5 11
s 7 18
s 5 12
s 7 19
s 5 13
s 8 15
s 5 14
s 8 16
s 5 15
s 8 17
s 1 14
s 8 18
s 1 15
s 8 19
s 1 16
s 9 15
s 1 17
s 9 16
s 1 18
s 9 17
s 1 19
s 9 18
s 2 14
s 9 19
s 2 15
s 6 16
s 2 16
s 6 17
s 2 17
s 6 18
s 2 18
s 6 19
s 2 19
s 7 16
s 3 14
s 7 17
s 3 15
s 7 18
s 3 16
s 7 19
s 3 17
s 8 16
s 3 18
s 8 17
s 3 19
s 8 18
s 4 14
s 8 19
s 4 15
s 9 16
s 4 16
s 9 17
s 4 17
s 9 18
s 4 18
s 9 19
s 4 19
s 4 19
s 5 14
s 5 19
s 5 15
s 6 19
s 5 16
s 7 19
s 5 17
s 8 19
s 5 18
s 9 19
s 5 19
s 6 14
s 6 15
s 6 16
s 6 17
s 6 18
s 6 19
s 7 14
s 7 15
s 7 16
s 7 17
s 7 18
s 7 19
s 1 15
s 1 16
s 1 17
s 1 18
s 1 19
s 2 15
s 2 16
s 2 17
s 2 18
s 2 19
s 3 15
s 3 16
s 3 17
s 3 18
s 3 19
s 4 15
s 4 16
s 4 17
s 4 18
s 4 19
s 5 15
s 5 16
s 5 17
s 5 18
s 5 19
s 6 15
s 6 16
s 6 17
s 6 18
s 6 19
s 1 16
s 1 17
s 1 18
s 1 19
s 2 16
s 2 17
s 2 18
s 2 19
s 3 16
s 3 17
s 3 18
s 3 19
s 4 16
s 4 17
s 4 18
s 4 19
s 5 16
s 5 17
s 5 18
s 5 19
s 1 19
s 2 19
s 3 19
s 4 19
s 5 19
s 6 19
s 7 19`;

console.log(data)
