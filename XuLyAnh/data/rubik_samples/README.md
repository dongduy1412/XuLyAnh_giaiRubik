# Rubik sample images

Place six front-facing Rubik face images here using these names:

```text
U.jpg  Up face
R.jpg  Right face
F.jpg  Front face
D.jpg  Down face
L.jpg  Left face
B.jpg  Back face
```

Capture guidance:

- Shoot each face as straight as possible.
- Make the Rubik face fill most of the frame.
- Keep lighting white and even.
- Avoid glare, shadows, and strong rotation.
- Keep the same physical cube orientation convention while capturing all six faces.

Run demo:

```powershell
python src/rubik_main.py --U data/rubik_samples/U.jpg --R data/rubik_samples/R.jpg --F data/rubik_samples/F.jpg --D data/rubik_samples/D.jpg --L data/rubik_samples/L.jpg --B data/rubik_samples/B.jpg --output results/rubik
```
