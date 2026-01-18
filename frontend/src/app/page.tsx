"use client";

import { useState, useRef, useEffect } from "react";
import { Upload, ArrowRight, ArrowDown, Loader2, ImageIcon, Zap, Sparkles, Bot, CheckCircle2 } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface CompareResult {
  query: string;
  original: {
    size: number[];
    tokens: number;
    detail_mode: string;
    prompt: string;
    base64_image: string;
  };
  optimized: {
    size: number[];
    tokens: number;
    detail_mode: string;
    prompt: string;
    base64_image: string;
    was_cropped: boolean;
    operations: string[];
  };
  savings: {
    tokens_saved: number;
    percent_saved: number;
    cost_saved: number;
    monthly_savings_10k: number;
  };
  intent: {
    detected: string;
    confidence: number;
    method: string;
  };
}

type AnimationStep =
  | "idle"
  | "analyzing"
  | "show-original-image"
  | "transforming-image"
  | "show-optimized-image"
  | "show-original-prompt"
  | "transforming-prompt"
  | "show-optimized-prompt"
  | "sending-to-model"
  | "waiting-response"
  | "show-results";

export default function Home() {
  const [query, setQuery] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [animationStep, setAnimationStep] = useState<AnimationStep>("idle");
  const [result, setResult] = useState<CompareResult | null>(null);
  const [originalResponse, setOriginalResponse] = useState<string | null>(null);
  const [optimizedResponse, setOptimizedResponse] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = () => setImagePreview(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

  const handleSubmit = async () => {
    if (!imageFile || !query.trim()) return;

    // Reset state
    setResult(null);
    setOriginalResponse(null);
    setOptimizedResponse(null);

    // Start animation sequence
    setAnimationStep("analyzing");
    await sleep(800);

    try {
      const formData = new FormData();
      formData.append("query", query);
      formData.append("image", imageFile);

      const response = await fetch(`${API_URL}/compare`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Failed to compare");

      const data: CompareResult = await response.json();
      setResult(data);

      // Animation sequence
      setAnimationStep("show-original-image");
      await sleep(1200);

      setAnimationStep("transforming-image");
      await sleep(1500);

      setAnimationStep("show-optimized-image");
      await sleep(1200);

      setAnimationStep("show-original-prompt");
      await sleep(1000);

      setAnimationStep("transforming-prompt");
      await sleep(1500);

      setAnimationStep("show-optimized-prompt");
      await sleep(1000);

      setAnimationStep("sending-to-model");
      await sleep(800);

      setAnimationStep("waiting-response");

      // Scroll to results
      resultsRef.current?.scrollIntoView({ behavior: "smooth" });

      // Call models in parallel
      const [origRes, optRes] = await Promise.all([
        fetch(`${API_URL}/call-model`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: data.original.prompt,
            base64_image: data.original.base64_image,
            detail_mode: data.original.detail_mode,
          }),
        }),
        fetch(`${API_URL}/call-model`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: data.optimized.prompt,
            base64_image: data.optimized.base64_image,
            detail_mode: data.optimized.detail_mode,
          }),
        }),
      ]);

      const origData = await origRes.json();
      const optData = await optRes.json();

      setOriginalResponse(origData.response || origData.error || "No response");
      setOptimizedResponse(optData.response || optData.error || "No response");

      setAnimationStep("show-results");

    } catch (error) {
      console.error("Error:", error);
      alert("Failed to process. Is the backend running on port 8000?");
      setAnimationStep("idle");
    }
  };

  const reset = () => {
    setAnimationStep("idle");
    setResult(null);
    setOriginalResponse(null);
    setOptimizedResponse(null);
    setQuery("");
    setImageFile(null);
    setImagePreview(null);
  };

  // Highlight removed words in prompt
  const renderPromptDiff = (original: string, optimized: string) => {
    const originalWords = original.split(/\s+/);
    const optimizedLower = optimized.toLowerCase();

    return (
      <span>
        {originalWords.map((word, i) => {
          const isKept = optimizedLower.includes(word.toLowerCase().replace(/[^a-z0-9]/g, ''));
          return (
            <span
              key={i}
              className={`${isKept ? 'text-green-600 font-medium' : 'text-red-400 line-through'} mr-1`}
            >
              {word}
            </span>
          );
        })}
      </span>
    );
  };

  const isProcessing = animationStep !== "idle" && animationStep !== "show-results";

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-gray-900" />
            <span className="font-semibold text-gray-900">TokenSqueeze</span>
          </div>
          <span className="text-sm text-gray-500">Powered by Token Company</span>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8">
        {/* Input Section */}
        {animationStep === "idle" && (
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-8 mb-8">
            <div className="text-center mb-8">
              <h1 className="text-2xl font-semibold text-gray-900 mb-2">
                See how much you can save
              </h1>
              <p className="text-gray-500">
                Upload an image and ask a question to see the optimization in action
              </p>
            </div>

            <div className="space-y-6">
              {/* Image Upload */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Image</label>
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className={`
                    relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer
                    transition-all duration-200
                    ${imagePreview
                      ? 'border-gray-300 bg-gray-50'
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'}
                  `}
                >
                  {imagePreview ? (
                    <img src={imagePreview} alt="Preview" className="max-h-48 mx-auto rounded-lg" />
                  ) : (
                    <div className="text-gray-400">
                      <Upload className="w-8 h-8 mx-auto mb-2" />
                      <p>Click to upload an image</p>
                    </div>
                  )}
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageChange}
                  className="hidden"
                />
              </div>

              {/* Query Input */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Your Question</label>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="e.g., Is the total on this receipt more than $25?"
                  rows={3}
                  className="w-full px-4 py-3 border border-gray-200 rounded-xl text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-200 focus:border-gray-300 resize-none"
                />
              </div>

              {/* Submit Button */}
              <button
                onClick={handleSubmit}
                disabled={!imageFile || !query.trim()}
                className="w-full py-4 bg-gray-900 text-white rounded-xl font-medium disabled:bg-gray-200 disabled:text-gray-400 disabled:cursor-not-allowed hover:bg-gray-800 transition-colors flex items-center justify-center gap-2"
              >
                <Sparkles className="w-5 h-5" />
                Optimize & Compare
              </button>
            </div>
          </div>
        )}

        {/* Animation Sequence */}
        {isProcessing && result && (
          <div className="space-y-8">
            {/* Step 1: Analyzing */}
            {animationStep === "analyzing" && (
              <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-12 text-center animate-pulse">
                <Loader2 className="w-12 h-12 mx-auto mb-4 text-gray-400 animate-spin" />
                <p className="text-lg font-medium text-gray-900">Analyzing your request...</p>
                <p className="text-gray-500 mt-1">Detecting intent and planning optimization</p>
              </div>
            )}

            {/* Step 2: Original Image */}
            {(animationStep === "show-original-image" ||
              animationStep === "transforming-image" ||
              animationStep === "show-optimized-image" ||
              animationStep === "show-original-prompt" ||
              animationStep === "transforming-prompt" ||
              animationStep === "show-optimized-prompt" ||
              animationStep === "sending-to-model" ||
              animationStep === "waiting-response") && (
              <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
                <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
                  <span className="font-medium text-gray-900">Image Optimization</span>
                  <span className="text-sm px-3 py-1 bg-gray-100 rounded-full text-gray-600">
                    {result.intent.detected.replace(/_/g, " ")}
                  </span>
                </div>

                <div className="p-6">
                  <div className="flex items-center gap-6">
                    {/* Original Image */}
                    <div className={`flex-1 text-center transition-all duration-500 ${
                      animationStep === "show-original-image" ? "scale-100 opacity-100" : "scale-95 opacity-60"
                    }`}>
                      <div className="text-sm text-gray-500 mb-2">Original</div>
                      <div className="bg-gray-50 rounded-xl p-4 inline-block">
                        <img
                          src={`data:image/jpeg;base64,${result.original.base64_image}`}
                          alt="Original"
                          className="max-h-40 rounded"
                        />
                      </div>
                      <div className="mt-2 text-xs text-gray-400">
                        {result.original.size[0]} × {result.original.size[1]}
                      </div>
                      <div className="mt-1 text-sm font-medium text-gray-600">
                        {result.original.tokens} tokens
                      </div>
                    </div>

                    {/* Arrow */}
                    <div className={`flex-shrink-0 transition-all duration-500 ${
                      animationStep === "transforming-image" ? "scale-125 text-gray-900" : "text-gray-300"
                    }`}>
                      <ArrowRight className="w-8 h-8" />
                    </div>

                    {/* Optimized Image */}
                    <div className={`flex-1 text-center transition-all duration-500 ${
                      animationStep === "show-optimized-image" ||
                      animationStep === "show-original-prompt" ||
                      animationStep === "transforming-prompt" ||
                      animationStep === "show-optimized-prompt" ||
                      animationStep === "sending-to-model" ||
                      animationStep === "waiting-response"
                        ? "scale-100 opacity-100"
                        : "scale-95 opacity-30"
                    }`}>
                      <div className="text-sm text-gray-500 mb-2">Optimized</div>
                      <div className="bg-green-50 rounded-xl p-4 inline-block border-2 border-green-200">
                        <img
                          src={`data:image/jpeg;base64,${result.optimized.base64_image}`}
                          alt="Optimized"
                          className="max-h-40 rounded"
                        />
                      </div>
                      <div className="mt-2 text-xs text-gray-400">
                        {result.optimized.size[0]} × {result.optimized.size[1]}
                        {result.optimized.was_cropped && " · cropped"}
                      </div>
                      <div className="mt-1 text-sm font-medium text-green-600">
                        {result.optimized.tokens} tokens
                      </div>
                    </div>
                  </div>

                  {/* Operations */}
                  {(animationStep === "show-optimized-image" ||
                    animationStep === "show-original-prompt" ||
                    animationStep === "transforming-prompt" ||
                    animationStep === "show-optimized-prompt" ||
                    animationStep === "sending-to-model" ||
                    animationStep === "waiting-response") && (
                    <div className="mt-4 flex flex-wrap gap-2 justify-center animate-fade-in">
                      {result.optimized.operations.map((op, i) => (
                        <span key={i} className="px-3 py-1 bg-gray-100 rounded-full text-xs text-gray-600">
                          {op}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Step 3: Prompt Optimization */}
            {(animationStep === "show-original-prompt" ||
              animationStep === "transforming-prompt" ||
              animationStep === "show-optimized-prompt" ||
              animationStep === "sending-to-model" ||
              animationStep === "waiting-response") && (
              <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
                <div className="px-6 py-4 border-b border-gray-100">
                  <span className="font-medium text-gray-900">Prompt Compression</span>
                  <span className="text-sm text-gray-500 ml-2">(Token Company API)</span>
                </div>

                <div className="p-6">
                  {/* Original Prompt */}
                  <div className={`transition-all duration-500 ${
                    animationStep === "show-original-prompt" ? "opacity-100" : "opacity-60"
                  }`}>
                    <div className="text-sm text-gray-500 mb-2">Original Prompt</div>
                    <div className="bg-gray-50 rounded-xl p-4 text-gray-700">
                      {animationStep === "transforming-prompt" ||
                       animationStep === "show-optimized-prompt" ||
                       animationStep === "sending-to-model" ||
                       animationStep === "waiting-response"
                        ? renderPromptDiff(result.original.prompt, result.optimized.prompt)
                        : result.original.prompt
                      }
                    </div>
                  </div>

                  {/* Arrow */}
                  <div className={`flex justify-center my-4 transition-all duration-500 ${
                    animationStep === "transforming-prompt" ? "scale-125 text-gray-900" : "text-gray-300"
                  }`}>
                    <ArrowDown className="w-6 h-6" />
                  </div>

                  {/* Optimized Prompt */}
                  <div className={`transition-all duration-500 ${
                    animationStep === "show-optimized-prompt" ||
                    animationStep === "sending-to-model" ||
                    animationStep === "waiting-response"
                      ? "opacity-100"
                      : "opacity-30"
                  }`}>
                    <div className="text-sm text-gray-500 mb-2">Compressed Prompt</div>
                    <div className="bg-green-50 rounded-xl p-4 text-green-700 border-2 border-green-200">
                      {result.optimized.prompt}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Step 4: Sending to Model */}
            {(animationStep === "sending-to-model" || animationStep === "waiting-response") && (
              <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-8 text-center">
                <div className="flex items-center justify-center gap-8">
                  <div className="text-center">
                    <div className="w-16 h-16 bg-gray-100 rounded-2xl flex items-center justify-center mb-2">
                      <ImageIcon className="w-8 h-8 text-gray-400" />
                    </div>
                    <div className="text-xs text-gray-500">Original</div>
                    <div className="text-sm font-medium">{result.original.tokens} tokens</div>
                  </div>

                  <div className={`transition-all duration-300 ${
                    animationStep === "waiting-response" ? "animate-pulse" : ""
                  }`}>
                    <ArrowRight className="w-6 h-6 text-gray-400" />
                  </div>

                  <div className="text-center">
                    <div className={`w-20 h-20 bg-gray-900 rounded-2xl flex items-center justify-center mb-2 ${
                      animationStep === "waiting-response" ? "animate-pulse" : ""
                    }`}>
                      <Bot className="w-10 h-10 text-white" />
                    </div>
                    <div className="text-sm font-medium">GPT-4o</div>
                  </div>

                  <div className={`transition-all duration-300 ${
                    animationStep === "waiting-response" ? "animate-pulse" : ""
                  }`}>
                    <ArrowRight className="w-6 h-6 text-gray-400" />
                  </div>

                  <div className="text-center">
                    <div className="w-16 h-16 bg-green-100 rounded-2xl flex items-center justify-center mb-2 border-2 border-green-200">
                      <ImageIcon className="w-8 h-8 text-green-600" />
                    </div>
                    <div className="text-xs text-gray-500">Optimized</div>
                    <div className="text-sm font-medium text-green-600">{result.optimized.tokens} tokens</div>
                  </div>
                </div>

                {animationStep === "waiting-response" && (
                  <div className="mt-6 flex items-center justify-center gap-2 text-gray-500">
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Waiting for responses...</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Final Results */}
        {animationStep === "show-results" && result && (
          <div ref={resultsRef} className="space-y-6 animate-fade-in">
            {/* Savings Banner */}
            <div className={`rounded-2xl p-6 ${
              result.savings.tokens_saved > 0
                ? 'bg-gradient-to-r from-green-500 to-emerald-500'
                : 'bg-gradient-to-r from-yellow-500 to-orange-500'
            }`}>
              <div className="text-center text-white">
                <div className="text-4xl font-bold mb-1">
                  {result.savings.tokens_saved > 0
                    ? `${result.savings.percent_saved.toFixed(0)}% Saved`
                    : `${Math.abs(result.savings.percent_saved).toFixed(0)}% More`
                  }
                </div>
                <div className="text-white/80">
                  {Math.abs(result.savings.tokens_saved)} tokens {result.savings.tokens_saved > 0 ? 'saved' : 'used'} ·
                  ${Math.abs(result.savings.monthly_savings_10k).toFixed(0)}/month at 10K requests/day
                </div>
              </div>
            </div>

            {/* Side by Side Responses */}
            <div className="grid md:grid-cols-2 gap-4">
              {/* Original Response */}
              <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
                <div className="px-5 py-4 bg-gray-50 border-b border-gray-200 flex items-center justify-between">
                  <span className="font-medium text-gray-700">Original</span>
                  <span className="text-sm text-gray-500">{result.original.tokens} tokens</span>
                </div>
                <div className="p-5">
                  <img
                    src={`data:image/jpeg;base64,${result.original.base64_image}`}
                    alt="Original"
                    className="w-full h-32 object-contain bg-gray-50 rounded-lg mb-4"
                  />
                  <div className="text-xs text-gray-400 mb-3">
                    {result.original.size[0]} × {result.original.size[1]} · {result.original.detail_mode} detail
                  </div>
                  <div className="border-t border-gray-100 pt-4">
                    <div className="text-xs font-medium text-gray-500 mb-2">Response:</div>
                    <div className="text-sm text-gray-700 bg-gray-50 rounded-lg p-3">
                      {originalResponse}
                    </div>
                  </div>
                </div>
              </div>

              {/* Optimized Response */}
              <div className="bg-white rounded-2xl shadow-sm border-2 border-green-200 overflow-hidden">
                <div className="px-5 py-4 bg-green-50 border-b border-green-200 flex items-center justify-between">
                  <span className="font-medium text-green-700">Optimized</span>
                  <span className="text-sm text-green-600">{result.optimized.tokens} tokens</span>
                </div>
                <div className="p-5">
                  <img
                    src={`data:image/jpeg;base64,${result.optimized.base64_image}`}
                    alt="Optimized"
                    className="w-full h-32 object-contain bg-gray-50 rounded-lg mb-4"
                  />
                  <div className="text-xs text-gray-400 mb-3">
                    {result.optimized.size[0]} × {result.optimized.size[1]} · {result.optimized.detail_mode} detail
                    {result.optimized.was_cropped && " · smart cropped"}
                  </div>
                  <div className="border-t border-gray-100 pt-4">
                    <div className="text-xs font-medium text-gray-500 mb-2">Response:</div>
                    <div className="text-sm text-gray-700 bg-green-50 rounded-lg p-3">
                      {optimizedResponse}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Conclusion */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 text-center">
              <CheckCircle2 className="w-10 h-10 mx-auto mb-3 text-green-500" />
              <div className="text-lg font-medium text-gray-900 mb-1">
                Same answer, {result.savings.tokens_saved > 0 ? 'fewer' : 'more'} tokens
              </div>
              <p className="text-gray-500 text-sm">
                The optimized version produced the same result
                {result.savings.tokens_saved > 0
                  ? ` while using ${result.savings.percent_saved.toFixed(0)}% fewer tokens.`
                  : ` but required more detail for accurate text extraction.`
                }
              </p>
            </div>

            {/* Try Again */}
            <button
              onClick={reset}
              className="w-full py-4 bg-gray-100 text-gray-700 rounded-xl font-medium hover:bg-gray-200 transition-colors"
            >
              Try Another Image
            </button>
          </div>
        )}
      </main>

      <style jsx global>{`
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fade-in 0.5s ease-out;
        }
      `}</style>
    </div>
  );
}
