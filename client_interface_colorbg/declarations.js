const serverurl = "http://127.0.0.1:8000";

const randomColor = document.getElementById("randomColor");
const remove_last_example = document.getElementById("remove_last_example");
const save_progress = document.getElementById("save");
const send_example = document.getElementById("sendE");
const lr = document.getElementById("lr");
const Tsamples = document.getElementById("Tsamples");
const predict = document.getElementById("predict");
const train = document.getElementById("train");
const text = document.getElementById("text");
const bginput = document.getElementById("bginput");
const bgoutput = document.getElementById("bgoutput");

const ctx = document.getElementById("myChart").getContext("2d");

const initialData = {
  labels: [],
  datasets: [
    {
      label: "Cost evolution",
      data: [],
      backgroundColor: "rgba(75, 192, 192, 0.2)",
      borderColor: "rgba(75, 192, 192, 1)",
      borderWidth: 1,
    },
  ],
};
const options = {
  scales: {
    y: {
      beginAtZero: true,
    },
  },
};
const myChart = new Chart(ctx, {
  type: "line",
  data: initialData,
  options: options,
});
