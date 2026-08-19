import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "next-themes";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";

import { ModelRegistryProvider } from "@/context/ModelRegistryContext";
import { PlaybackProvider } from "@/contexts/PlaybackContext";
import { ModelDownloadBanner } from "@/components/layout/ModelDownloadBanner";

const queryClient = new QueryClient();

const App = () => (
  // SRS §3.6.6: dark mode follows system preference with a manual override.
  // `attribute="class"` matches tailwind.config.ts's `darkMode: ["class"]`.
  <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
    <QueryClientProvider client={queryClient}>
      <ModelRegistryProvider>
        <PlaybackProvider>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <ModelDownloadBanner />
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<Index />} />
              {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </BrowserRouter>
        </TooltipProvider>
        </PlaybackProvider>
      </ModelRegistryProvider>
    </QueryClientProvider>
  </ThemeProvider>
);

export default App;
