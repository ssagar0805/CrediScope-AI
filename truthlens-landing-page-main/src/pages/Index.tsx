import React from "react";
import TruthLensHeader from "@/components/TruthLensHeader";
import HeroSection from "@/components/HeroSection"; 
import HowItWorksSection from "@/components/HowItWorksSection";
import TrendsSection from "@/components/TrendsSection";
import EducationSection from "@/components/EducationSection";
import FinalCTASection from "@/components/FinalCTASection";
import TruthLensFooter from "@/components/TruthLensFooter";

const Index = () => {
  return (
    <div id="truthlens-root" className="min-h-screen flex flex-col">
      <TruthLensHeader />
      <main className="flex-1">
        <HeroSection />
        <HowItWorksSection />
        <TrendsSection />
        <EducationSection />
        <FinalCTASection />
      </main>
      <TruthLensFooter />
    </div>
  );
};

export default Index;
