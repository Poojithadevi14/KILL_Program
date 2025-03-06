function predictChurn() {
    const features = [
        parseFloat(document.getElementById("credit_score").value),
        parseFloat(document.getElementById("geography").value),
        parseFloat(document.getElementById("age").value),
        parseFloat(document.getElementById("tenure").value),
        parseFloat(document.getElementById("balance").value),
        parseFloat(document.getElementById("products").value),
        parseFloat(document.getElementById("credit_card").value),
        parseFloat(document.getElementById("active_member").value),
        parseFloat(document.getElementById("salary").value),
        parseFloat(document.getElementById("gender").value)
    ];

    fetch('/predict', {
        method: 'POST',
        body: JSON.stringify({ features: features }),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerHTML = "Prediction: " + data.prediction;
    })
    .catch(error => console.error("Error:", error));
}
