const express = require("express");
const http = require("http");
const { Server } = require("socket.io");

const app = express();
const server = http.createServer(app);

const io = new Server(server, {
  cors: { origin: "*" }
});

let robotSocket = null;
let policeSocket = null;

io.on("connection", (socket) => {
  console.log("🔌 Connected:", socket.id);

  socket.on("register", (role) => {
    console.log("Register:", role);

    if (role === "robot") {
      robotSocket = socket;
      console.log("🤖 Robot connected");
    }

    if (role === "police") {
      policeSocket = socket;
      console.log("👮 Police connected");
    }
  });

  socket.on("ready", () => {
    console.log("👮 Police ready");

    if (robotSocket) {
      console.log("➡️ Sending start-call to robot");
      robotSocket.emit("start-call");
    }
  });

  socket.on("offer", (offer) => {
    console.log("📤 Offer → police");
    policeSocket?.emit("offer", offer);
  });

  socket.on("answer", (answer) => {
    console.log("📤 Answer → robot");
    robotSocket?.emit("answer", answer);
  });

  socket.on("ice-candidate", (candidate) => {
    if (socket === robotSocket) {
      policeSocket?.emit("ice-candidate", candidate);
    } else {
      robotSocket?.emit("ice-candidate", candidate);
    }
  });

  socket.on("disconnect", () => {
    console.log("❌ Disconnected:", socket.id);

    if (socket === robotSocket) robotSocket = null;
    if (socket === policeSocket) policeSocket = null;
  });
});

server.listen(3000, () => {
  console.log("🚀 Server running on http://127.0.0.1:3000");
});