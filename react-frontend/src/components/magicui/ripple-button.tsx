import React from "react";
import { cn } from "../../lib/utils";

interface RippleButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  rippleColor?: string;
  className?: string;
}

const RippleButton = React.forwardRef<HTMLButtonElement, RippleButtonProps>(
  ({ children, className, rippleColor = "#ffffff", ...props }, ref) => {
    return (
      <button
        className={cn(
          "relative overflow-hidden rounded-md bg-primary px-6 py-3 text-primary-foreground",
          "transform-gpu transition-all duration-300 ease-in-out",
          "hover:shadow-lg active:scale-95",
          "[&_[data-ripple]]:absolute [&_[data-ripple]]:inset-0",
          "[&_[data-ripple]]:rounded-full [&_[data-ripple]]:bg-white/25",
          "[&_[data-ripple]]:opacity-0 [&_[data-ripple]]:scale-0",
          "[&_[data-ripple]]:animate-ripple",
          className
        )}
        style={
          {
            "--ripple-color": rippleColor,
          } as React.CSSProperties
        }
        ref={ref}
        onClick={(e) => {
          const button = e.currentTarget;
          const rect = button.getBoundingClientRect();
          const size = Math.max(rect.width, rect.height);
          const x = e.clientX - rect.left - size / 2;
          const y = e.clientY - rect.top - size / 2;

          const ripple = document.createElement("span");
          ripple.setAttribute("data-ripple", "");
          ripple.style.width = ripple.style.height = size + "px";
          ripple.style.left = x + "px";
          ripple.style.top = y + "px";
          ripple.style.backgroundColor = rippleColor;

          button.appendChild(ripple);

          // Remove ripple after animation
          setTimeout(() => {
            ripple.remove();
          }, 600);

          // Call original onClick if provided
          if (props.onClick) {
            props.onClick(e);
          }
        }}
        {...props}
      >
        {children}
      </button>
    );
  }
);

RippleButton.displayName = "RippleButton";

export { RippleButton }; 