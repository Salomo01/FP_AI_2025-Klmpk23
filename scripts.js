document.getElementById('upload-form').addEventListener('submit', async function(e) {
  e.preventDefault();

  const image1 = document.getElementById('image1').files[0];
  const image2 = document.getElementById('image2').files[0];
  const resultDiv = document.getElementById('result');

  if (!image1 || !image2) {
    resultDiv.textContent = 'Please select both images.';
    return;
  }

  const formData = new FormData();
  formData.append('image1', image1);
  formData.append('image2', image2);

  resultDiv.textContent = 'Predicting…';

  try {
    const response = await fetch('/upload', {
      method: 'POST',
      body: formData
    });
    const data = await response.json();

    if (response.ok) {
      if (data.verified) {
        resultDiv.textContent = `✅ Faces match! Distance: ${data.distance.toFixed(4)}`;
      } else {
        resultDiv.textContent = `❌ Faces do not match. Distance: ${data.distance.toFixed(4)}`;
      }
    } else {
      resultDiv.textContent = `Error: ${data.error}`;
    }
  } catch (err) {
    resultDiv.textContent = `Error: ${err.message}`;
  }
});
