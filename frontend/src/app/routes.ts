import { createBrowserRouter } from "react-router";
import LandingPage from "./components/LandingPage";
import ChatbotPage from "./components/ChatbotPage";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: LandingPage,
  },
  {
    path: "/chat",
    Component: ChatbotPage,
  },
]);
