"""
Cinematic Gaming-Grade Traffic Simulation Renderer.
Features:
- High-contrast Cyber-City / Smart-City visual theme with grid background & neon curbs
- Detailed vehicle models with dynamic forward headlight cones & illuminated red brake halos
- Crosswalk zebra stripes, lane arrows, and textured asphalt
- Futuristic illuminated traffic signal gantries with triple-layer radial halo glow
- Large Tactical Gaming HUD: Top Command Banner, Efficiency Grade, and 4 Large Approach Radar Cards
"""
import math
import os
import pygame
import numpy as np
from typing import Dict, List, Optional, Tuple
from utils.config import Config, Direction, SignalPhase
from environment.vehicle import Vehicle
from environment.traffic_intersection import TrafficIntersection


class IntersectionRenderer:
    """Renders a cinematic, game-style 4-way intersection simulation."""

    def __init__(self, config: Config | None = None, is_headless: bool = True):
        self.config = config or Config()
        self.is_headless = is_headless
        self.width = self.config.SCREEN_WIDTH
        self.height = self.config.SCREEN_HEIGHT

        if not pygame.get_init():
            if is_headless:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.init()

        if not pygame.font.get_init():
            pygame.font.init()

        self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.glow_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # High-visibility fonts
        self.font_tiny = pygame.font.SysFont("Trebuchet MS", 11, bold=True)
        self.font_main = pygame.font.SysFont("Trebuchet MS", 13, bold=True)
        self.font_bold = pygame.font.SysFont("Trebuchet MS", 15, bold=True)
        self.font_large = pygame.font.SysFont("Trebuchet MS", 18, bold=True)
        self.font_huge = pygame.font.SysFont("Impact", 24)

        # Animation tick
        self.anim_frame = 0

    def render(
        self,
        intersection: TrafficIntersection,
        agent_name: str = "RL Agent",
        metrics: Optional[Dict[str, any]] = None,
    ) -> pygame.Surface:
        """Draws the full cinematic gaming-style frame."""
        self.anim_frame += 1

        # 1. Dark Tech Terrain & City Grid
        self._draw_cyber_terrain()

        # 2. Road Geometry, Curbs & Crosswalks
        self._draw_roads_and_crosswalks()

        # 3. Pavement Markings & Directional Arrows
        self._draw_lane_markings()

        # 4. Traffic Signal Gantries (Futuristic Glowing Towers)
        self._draw_signal_gantries(intersection.signal_controller)

        # 5. Vehicles (Sleek Automotive Sprites with Headlight Beams & Brake Halos)
        self._draw_gaming_vehicles(intersection.active_vehicles)

        # 6. Large Tactical Gaming HUD (Top Command Banner & Bottom Approach Radars)
        self._draw_tactical_hud(intersection, agent_name, metrics)

        return self.surface

    def get_rgb_array(self) -> np.ndarray:
        """Converts surface to RGB numpy array for Streamlit."""
        rgb_surf = pygame.surfarray.array3d(self.surface)
        return np.transpose(rgb_surf, (1, 0, 2))

    def _draw_cyber_terrain(self):
        """Draws a deep obsidian cyber-city grid background."""
        self.surface.fill((13, 17, 23))  # Deep midnight obsidian

        # Subtle glowing tech grid lines
        grid_spacing = 35
        for x in range(0, self.width, grid_spacing):
            pygame.draw.line(self.surface, (22, 30, 46), (x, 0), (x, self.height), 1)
        for y in range(0, self.height, grid_spacing):
            pygame.draw.line(self.surface, (22, 30, 46), (0, y), (self.width, y), 1)

    def _draw_roads_and_crosswalks(self):
        """Renders textured asphalt, sidewalk curbs, and pedestrian crosswalk zebra stripes."""
        c = self.config
        cx, cy = c.CENTER_X, c.CENTER_Y
        rw = c.ROAD_WIDTH

        # 1. Sidewalk Curbs with Neon Edge Accent
        curb_pad = 8
        curb_col = (30, 41, 59)
        neon_border = (56, 189, 248)  # Cyan glow edge

        # Vertical road curb
        v_curb = pygame.Rect(cx - rw // 2 - curb_pad, 0, rw + curb_pad * 2, self.height)
        pygame.draw.rect(self.surface, curb_col, v_curb)
        pygame.draw.rect(self.surface, neon_border, v_curb, 1)

        # Horizontal road curb
        h_curb = pygame.Rect(0, cy - rw // 2 - curb_pad, self.width, rw + curb_pad * 2)
        pygame.draw.rect(self.surface, curb_col, h_curb)
        pygame.draw.rect(self.surface, neon_border, h_curb, 1)

        # 2. Main Asphalt Roadways
        road_col = (28, 33, 44)  # High-grade dark asphalt
        v_road = pygame.Rect(cx - rw // 2, 0, rw, self.height)
        h_road = pygame.Rect(0, cy - rw // 2, self.width, rw)
        pygame.draw.rect(self.surface, road_col, v_road)
        pygame.draw.rect(self.surface, road_col, h_road)

        # 3. Intersection Core Junction
        box_rect = pygame.Rect(cx - rw // 2, cy - rw // 2, rw, rw)
        pygame.draw.rect(self.surface, (24, 29, 39), box_rect)
        pygame.draw.rect(self.surface, (51, 65, 85), box_rect, 1)

        # 4. Crosswalk Zebra Stripes at all 4 Approaches
        stripe_color = (203, 213, 225)
        stripe_w = 20
        stripe_h = 7
        gap = 5

        # North Crosswalk (between Y=255 and Y=275)
        for x in range(cx - rw // 2 + 5, cx + rw // 2 - 5, stripe_h + gap):
            pygame.draw.rect(self.surface, stripe_color, (x, int(c.STOP_LINE_NORTH) - 22, stripe_h, stripe_w))

        # South Crosswalk (between Y=425 and Y=445)
        for x in range(cx - rw // 2 + 5, cx + rw // 2 - 5, stripe_h + gap):
            pygame.draw.rect(self.surface, stripe_color, (x, int(c.STOP_LINE_SOUTH) + 4, stripe_h, stripe_w))

        # West Crosswalk (between X=255 and X=275)
        for y in range(cy - rw // 2 + 5, cy + rw // 2 - 5, stripe_h + gap):
            pygame.draw.rect(self.surface, stripe_color, (int(c.STOP_LINE_WEST) - 22, y, stripe_w, stripe_h))

        # East Crosswalk (between X=425 and X=445)
        for y in range(cy - rw // 2 + 5, cy + rw // 2 - 5, stripe_h + gap):
            pygame.draw.rect(self.surface, stripe_color, (int(c.STOP_LINE_EAST) + 4, y, stripe_w, stripe_h))

    def _draw_lane_markings(self):
        """Draws double yellow center lines, lane dashes, and solid stop lines."""
        c = self.config
        cx, cy = c.CENTER_X, c.CENTER_Y
        rw = c.ROAD_WIDTH
        lw = c.LANE_WIDTH

        yellow_line = (245, 158, 11)
        dash_color = (226, 232, 240)
        stop_line_col = (255, 255, 255)

        # 1. Double Yellow Dividing Lines
        pygame.draw.line(self.surface, yellow_line, (cx - 2, 0), (cx - 2, cy - rw // 2), 2)
        pygame.draw.line(self.surface, yellow_line, (cx + 2, 0), (cx + 2, cy - rw // 2), 2)
        pygame.draw.line(self.surface, yellow_line, (cx - 2, cy + rw // 2), (cx - 2, self.height), 2)
        pygame.draw.line(self.surface, yellow_line, (cx + 2, cy + rw // 2), (cx + 2, self.height), 2)

        pygame.draw.line(self.surface, yellow_line, (0, cy - 2), (cx - rw // 2, cy - 2), 2)
        pygame.draw.line(self.surface, yellow_line, (0, cy + 2), (cx - rw // 2, cy + 2), 2)
        pygame.draw.line(self.surface, yellow_line, (cx + rw // 2, cy - 2), (self.width, cy - 2), 2)
        pygame.draw.line(self.surface, yellow_line, (cx + rw // 2, cy + 2), (self.width, cy + 2), 2)

        # 2. Dashed Lane Divisors
        def _draw_dashed_v(x, y_start, y_end):
            step = 1 if y_end > y_start else -1
            y = y_start
            while (y < y_end if step > 0 else y > y_end):
                next_y = y + (16 * step)
                pygame.draw.line(self.surface, dash_color, (x, y), (x, next_y), 2)
                y += 26 * step

        def _draw_dashed_h(y, x_start, x_end):
            step = 1 if x_end > x_start else -1
            x = x_start
            while (x < x_end if step > 0 else x > x_end):
                next_x = x + (16 * step)
                pygame.draw.line(self.surface, dash_color, (x, y), (next_x, y), 2)
                x += 26 * step

        _draw_dashed_v(cx + lw, 0, cy - rw // 2 - 25)
        _draw_dashed_v(cx - lw, 0, cy - rw // 2 - 25)
        _draw_dashed_v(cx - lw, self.height, cy + rw // 2 + 25)
        _draw_dashed_v(cx + lw, self.height, cy + rw // 2 + 25)

        _draw_dashed_h(cy + lw, 0, cx - rw // 2 - 25)
        _draw_dashed_h(cy - lw, 0, cx - rw // 2 - 25)
        _draw_dashed_h(cy - lw, self.width, cx + rw // 2 + 25)
        _draw_dashed_h(cy + lw, self.width, cx + rw // 2 + 25)

        # 3. Glowing Solid Stop Bars
        pygame.draw.line(self.surface, stop_line_col, (cx, c.STOP_LINE_NORTH), (cx + rw // 2, c.STOP_LINE_NORTH), 4)
        pygame.draw.line(self.surface, stop_line_col, (cx - rw // 2, c.STOP_LINE_SOUTH), (cx, c.STOP_LINE_SOUTH), 4)
        pygame.draw.line(self.surface, stop_line_col, (c.STOP_LINE_EAST, cy - rw // 2), (c.STOP_LINE_EAST, cy), 4)
        pygame.draw.line(self.surface, stop_line_col, (c.STOP_LINE_WEST, cy), (c.STOP_LINE_WEST, cy + rw // 2), 4)

        # 4. Lane Direction Arrows on Inbound Approaches
        arrow_col = (148, 163, 184)
        self._draw_arrow((cx + lw // 2, cy - rw // 2 - 70), "DOWN", arrow_col)
        self._draw_arrow((cx + lw + lw // 2, cy - rw // 2 - 70), "DOWN", arrow_col)
        self._draw_arrow((cx - lw // 2, cy + rw // 2 + 70), "UP", arrow_col)
        self._draw_arrow((cx - lw - lw // 2, cy + rw // 2 + 70), "UP", arrow_col)
        self._draw_arrow((cx - rw // 2 - 70, cy + lw // 2), "RIGHT", arrow_col)
        self._draw_arrow((cx - rw // 2 - 70, cy + lw + lw // 2), "RIGHT", arrow_col)
        self._draw_arrow((cx + rw // 2 + 70, cy - lw // 2), "LEFT", arrow_col)
        self._draw_arrow((cx + rw // 2 + 70, cy - lw - lw // 2), "LEFT", arrow_col)

    def _draw_arrow(self, center: Tuple[int, int], direction: str, color: Tuple[int, int, int]):
        """Draws clean road arrow markings."""
        x, y = center
        if direction == "DOWN":
            pts = [(x, y + 10), (x - 6, y + 2), (x - 2, y + 2), (x - 2, y - 10), (x + 2, y - 10), (x + 2, y + 2), (x + 6, y + 2)]
        elif direction == "UP":
            pts = [(x, y - 10), (x - 6, y - 2), (x - 2, y - 2), (x - 2, y + 10), (x + 2, y + 10), (x + 2, y - 2), (x + 6, y - 2)]
        elif direction == "RIGHT":
            pts = [(x + 10, y), (x + 2, y - 6), (x + 2, y - 2), (x - 10, y - 2), (x - 10, y + 2), (x + 2, y + 2), (x + 2, y + 6)]
        else:  # LEFT
            pts = [(x - 10, y), (x - 2, y - 6), (x - 2, y - 2), (x + 10, y - 2), (x + 10, y + 2), (x - 2, y + 2), (x - 2, y + 6)]
        pygame.draw.polygon(self.surface, color, pts)

    def _draw_signal_gantries(self, signal_controller):
        """Draws large, prominent sci-fi traffic light gantries with radial halos."""
        c = self.config
        cx, cy = c.CENTER_X, c.CENTER_Y
        rw = c.ROAD_WIDTH

        # 4 Gantry positions placed prominently near stop lines
        gantry_coords = {
            Direction.NORTH: (cx + rw // 2 + 16, cy - rw // 2 - 45, "V"),
            Direction.SOUTH: (cx - rw // 2 - 44, cy + rw // 2 + 15, "V"),
            Direction.EAST: (cx + rw // 2 + 15, cy + rw // 2 + 16, "H"),
            Direction.WEST: (cx - rw // 2 - 45, cy - rw // 2 - 44, "H"),
        }

        pulse = (math.sin(self.anim_frame * 0.15) + 1.0) * 0.5  # 0.0 to 1.0

        for direction, (x, y, orient) in gantry_coords.items():
            is_green = signal_controller.is_green(direction)
            is_yellow = signal_controller.is_yellow(direction)
            is_red = signal_controller.is_red(direction)

            housing_col = (15, 23, 42)
            border_col = (71, 85, 105)

            if orient == "V":
                h_rect = pygame.Rect(x, y, 26, 68)
                pygame.draw.rect(self.surface, housing_col, h_rect, border_radius=6)
                pygame.draw.rect(self.surface, border_col, h_rect, 2, border_radius=6)

                # Bulbs: Red (top), Yellow (mid), Green (bot)
                r_pos = (x + 13, y + 14)
                y_pos = (x + 13, y + 34)
                g_pos = (x + 13, y + 54)

                self._draw_glowing_bulb(r_pos, (239, 68, 68), is_red, pulse)
                self._draw_glowing_bulb(y_pos, (245, 158, 11), is_yellow, pulse)
                self._draw_glowing_bulb(g_pos, (16, 185, 129), is_green, pulse)

            else:  # Horizontal
                h_rect = pygame.Rect(x, y, 68, 26)
                pygame.draw.rect(self.surface, housing_col, h_rect, border_radius=6)
                pygame.draw.rect(self.surface, border_col, h_rect, 2, border_radius=6)

                r_pos = (x + 14, y + 13)
                y_pos = (x + 34, y + 13)
                g_pos = (x + 54, y + 13)

                self._draw_glowing_bulb(r_pos, (239, 68, 68), is_red, pulse)
                self._draw_glowing_bulb(y_pos, (245, 158, 11), is_yellow, pulse)
                self._draw_glowing_bulb(g_pos, (16, 185, 129), is_green, pulse)

    def _draw_glowing_bulb(self, center: Tuple[int, int], color: Tuple[int, int, int], is_active: bool, pulse: float):
        """Draws individual multi-layer glowing lens."""
        cx, cy = center
        if is_active:
            # Outer radial glow
            glow_radius = int(14 + pulse * 4)
            halo_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(halo_surf, (*color, int(60 + pulse * 40)), (glow_radius, glow_radius), glow_radius)
            pygame.draw.circle(halo_surf, (*color, int(120 + pulse * 50)), (glow_radius, glow_radius), glow_radius - 4)
            self.surface.blit(halo_surf, (cx - glow_radius, cy - glow_radius))

            # Core bright bulb
            pygame.draw.circle(self.surface, color, (cx, cy), 8)
            # Center specular highlight
            pygame.draw.circle(self.surface, (255, 255, 255), (cx - 1, cy - 1), 3)
        else:
            # Off state (dim housing lens)
            pygame.draw.circle(self.surface, (45, 55, 72), (cx, cy), 7)
            pygame.draw.circle(self.surface, (25, 30, 40), (cx, cy), 5)

    def _draw_gaming_vehicles(self, vehicles: List[Vehicle]):
        """Renders stylized vehicles with forward headlight cones and red brake halos."""
        for v in vehicles:
            vx, vy = v.x, v.y
            vl, vw = v.length, v.width

            # 1. Forward Headlight Cones (projecting beam of light on asphalt)
            self._draw_headlight_beam(v)

            # 2. Main Vehicle Body (Aerodynamic rounded chassis with roof contrast)
            if v.direction in (Direction.NORTH, Direction.SOUTH):
                # Vertical
                body_rect = pygame.Rect(vx - vw / 2, vy - vl / 2, vw, vl)
                pygame.draw.rect(self.surface, (10, 15, 25), (vx - vw / 2 + 1, vy - vl / 2 + 2, vw, vl), border_radius=4)  # Shadow
                pygame.draw.rect(self.surface, v.color, body_rect, border_radius=4)
                pygame.draw.rect(self.surface, (255, 255, 255), body_rect, 1, border_radius=4)

                # Windshield & Roof
                ws_y = vy - 3 if v.direction == Direction.NORTH else vy + 1
                pygame.draw.rect(self.surface, (15, 23, 42), (vx - vw / 2 + 2, ws_y, vw - 4, 5), border_radius=2)
                # Roof highlight
                roof_y = vy + 3 if v.direction == Direction.NORTH else vy - 5
                pygame.draw.rect(self.surface, (30, 41, 59), (vx - vw / 2 + 3, roof_y, vw - 6, 4), border_radius=1)

                # Headlights (Yellow/White dots)
                hl_y = vy + vl / 2 - 2 if v.direction == Direction.NORTH else vy - vl / 2 + 2
                pygame.draw.circle(self.surface, (255, 255, 200), (int(vx - vw / 2 + 3), int(hl_y)), 2)
                pygame.draw.circle(self.surface, (255, 255, 200), (int(vx + vw / 2 - 3), int(hl_y)), 2)

                # Brake Lights / Tail Halos
                tl_y = vy - vl / 2 + 1 if v.direction == Direction.NORTH else vy + vl / 2 - 1
                if v.is_queued or v.speed < 2.0:
                    self._draw_brake_halo(int(vx - vw / 2 + 3), int(tl_y))
                    self._draw_brake_halo(int(vx + vw / 2 - 3), int(tl_y))
                else:
                    pygame.draw.circle(self.surface, (185, 28, 28), (int(vx - vw / 2 + 3), int(tl_y)), 2)
                    pygame.draw.circle(self.surface, (185, 28, 28), (int(vx + vw / 2 - 3), int(tl_y)), 2)

            else:
                # Horizontal
                body_rect = pygame.Rect(vx - vl / 2, vy - vw / 2, vl, vw)
                pygame.draw.rect(self.surface, (10, 15, 25), (vx - vl / 2 + 2, vy - vw / 2 + 1, vl, vw), border_radius=4)  # Shadow
                pygame.draw.rect(self.surface, v.color, body_rect, border_radius=4)
                pygame.draw.rect(self.surface, (255, 255, 255), body_rect, 1, border_radius=4)

                # Windshield
                ws_x = vx - 3 if v.direction == Direction.EAST else vx + 1
                pygame.draw.rect(self.surface, (15, 23, 42), (ws_x, vy - vw / 2 + 2, 5, vw - 4), border_radius=2)
                # Roof
                roof_x = vx + 3 if v.direction == Direction.EAST else vx - 5
                pygame.draw.rect(self.surface, (30, 41, 59), (roof_x, vy - vw / 2 + 3, 4, vw - 6), border_radius=1)

                hl_x = vx - vl / 2 + 2 if v.direction == Direction.EAST else vx + vl / 2 - 2
                pygame.draw.circle(self.surface, (255, 255, 200), (int(hl_x), int(vy - vw / 2 + 3)), 2)
                pygame.draw.circle(self.surface, (255, 255, 200), (int(hl_x), int(vy + vw / 2 - 3)), 2)

                tl_x = vx + vl / 2 - 1 if v.direction == Direction.EAST else vx - vl / 2 + 1
                if v.is_queued or v.speed < 2.0:
                    self._draw_brake_halo(int(tl_x), int(vy - vw / 2 + 3))
                    self._draw_brake_halo(int(tl_x), int(vy + vw / 2 - 3))
                else:
                    pygame.draw.circle(self.surface, (185, 28, 28), (int(tl_x), int(vy - vw / 2 + 3)), 2)
                    pygame.draw.circle(self.surface, (185, 28, 28), (int(tl_x), int(vy + vw / 2 - 3)), 2)

    def _draw_headlight_beam(self, v: Vehicle):
        """Draws semi-transparent light beam projection ahead of moving vehicles."""
        beam_len = 35
        beam_spread = 14
        vx, vy = v.x, v.y
        beam_surf = pygame.Surface((80, 80), pygame.SRCALPHA)
        beam_col = (255, 255, 220, 35)

        if v.direction == Direction.NORTH:
            pts = [(40, 20), (40 - beam_spread, 20 + beam_len), (40 + beam_spread, 20 + beam_len)]
            pygame.draw.polygon(beam_surf, beam_col, pts)
            self.surface.blit(beam_surf, (vx - 40, vy))
        elif v.direction == Direction.SOUTH:
            pts = [(40, 60), (40 - beam_spread, 60 - beam_len), (40 + beam_spread, 60 - beam_len)]
            pygame.draw.polygon(beam_surf, beam_col, pts)
            self.surface.blit(beam_surf, (vx - 40, vy - 60))
        elif v.direction == Direction.EAST:
            pts = [(60, 40), (60 - beam_len, 40 - beam_spread), (60 - beam_len, 40 + beam_spread)]
            pygame.draw.polygon(beam_surf, beam_col, pts)
            self.surface.blit(beam_surf, (vx - 60, vy - 40))
        elif v.direction == Direction.WEST:
            pts = [(20, 40), (20 + beam_len, 40 - beam_spread), (20 + beam_len, 40 + beam_spread)]
            pygame.draw.polygon(beam_surf, beam_col, pts)
            self.surface.blit(beam_surf, (vx, vy - 40))

    def _draw_brake_halo(self, x: int, y: int):
        """Draws vibrant red brake light corona flare when vehicle stops in queue."""
        halo_surf = pygame.Surface((18, 18), pygame.SRCALPHA)
        pygame.draw.circle(halo_surf, (239, 68, 68, 80), (9, 9), 8)
        pygame.draw.circle(halo_surf, (239, 68, 68, 180), (9, 9), 5)
        pygame.draw.circle(halo_surf, (255, 255, 255), (9, 9), 2)
        self.surface.blit(halo_surf, (x - 9, y - 9))

    def _draw_tactical_hud(
        self,
        intersection: TrafficIntersection,
        agent_name: str,
        metrics: Optional[Dict[str, any]],
    ):
        """Draws large, professional, high-impact gaming HUD cards."""
        # -------------------------------------------------------------
        # 1. TOP COMMAND BAR (Large Cyberpunk Glass Header)
        # -------------------------------------------------------------
        top_bar_rect = pygame.Rect(12, 10, self.width - 24, 56)
        glass_surf = pygame.Surface((top_bar_rect.width, top_bar_rect.height), pygame.SRCALPHA)
        glass_surf.fill((15, 23, 42, 235))
        self.surface.blit(glass_surf, (top_bar_rect.x, top_bar_rect.y))
        pygame.draw.rect(self.surface, (56, 189, 248), top_bar_rect, 2, border_radius=8)

        # Title & Controller Badge
        title_surf = self.font_huge.render("SMART CITY AI INTERSECTION", True, (248, 250, 252))
        self.surface.blit(title_surf, (24, 14))

        agent_badge_col = (16, 185, 129) if "Q-Learning" in agent_name or "DQN" in agent_name else (239, 68, 68)
        badge_text = f"MODE: {agent_name.upper()}"
        b_surf = self.font_bold.render(badge_text, True, agent_badge_col)
        self.surface.blit(b_surf, (24, 40))

        # Right side: Live Key Metrics readout
        if metrics:
            passed = metrics.get("total_passed", 0)
            avg_wait = metrics.get("avg_waiting_time", 0.0)
            q_total = metrics.get("total_queue", 0)

            # Rating Grade calculation
            grade = "A+" if q_total <= 3 and avg_wait < 3.0 else ("A" if q_total <= 6 else ("B" if q_total <= 10 else "C"))
            grade_color = (16, 185, 129) if grade in ("A+", "A") else ((245, 158, 11) if grade == "B" else (239, 68, 68))

            score_surf = self.font_huge.render(f"GRADE {grade}", True, grade_color)
            self.surface.blit(score_surf, (self.width - score_surf.get_width() - 28, 14))

            kpi_str = f"CLEARED: {passed} VEH   |   AVG WAIT: {avg_wait:.1f}s   |   QUEUED: {q_total}"
            kpi_surf = self.font_main.render(kpi_str, True, (226, 232, 240))
            self.surface.blit(kpi_surf, (self.width - kpi_surf.get_width() - 28, 42))

        # -------------------------------------------------------------
        # 2. BOTTOM TACTICAL RADAR (4 Large Approach Cards across screen)
        # -------------------------------------------------------------
        if metrics and "queue_lengths" in metrics:
            q_map = metrics["queue_lengths"]
            c_map = metrics["vehicle_counts"]
            signals = intersection.signal_controller.get_signal_states()

            approaches = [
                ("NORTH", "N-APPROACH", q_map.get("NORTH", 0), c_map.get("NORTH", 0), signals.get("NORTH", "RED")),
                ("SOUTH", "S-APPROACH", q_map.get("SOUTH", 0), c_map.get("SOUTH", 0), signals.get("SOUTH", "RED")),
                ("EAST", "E-APPROACH", q_map.get("EAST", 0), c_map.get("EAST", 0), signals.get("EAST", "RED")),
                ("WEST", "W-APPROACH", q_map.get("WEST", 0), c_map.get("WEST", 0), signals.get("WEST", "RED")),
            ]

            card_w = (self.width - 24 - 36) // 4  # 4 equal wide cards
            card_h = 58
            card_y = self.height - card_h - 12

            for i, (dir_key, label, q_len, count, sig_state) in enumerate(approaches):
                cx = 12 + i * (card_w + 12)
                self._draw_large_approach_card(cx, card_y, card_w, card_h, label, q_len, count, sig_state)

    def _draw_large_approach_card(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        title: str,
        queue_count: int,
        total_veh: int,
        signal_state: str,
    ):
        """Draws an individual large, prominent approach radar card."""
        card_rect = pygame.Rect(x, y, w, h)
        glass = pygame.Surface((w, h), pygame.SRCALPHA)
        glass.fill((15, 23, 42, 235))
        self.surface.blit(glass, (x, y))

        border_col = (16, 185, 129) if signal_state == "GREEN" else ((245, 158, 11) if signal_state == "YELLOW" else (239, 68, 68))
        pygame.draw.rect(self.surface, border_col, card_rect, 2, border_radius=6)

        t_surf = self.font_bold.render(title, True, (248, 250, 252))
        self.surface.blit(t_surf, (x + 10, y + 6))

        pill_txt = f"[{signal_state}]"
        p_surf = self.font_tiny.render(pill_txt, True, border_col)
        self.surface.blit(p_surf, (x + w - p_surf.get_width() - 8, y + 8))

        q_color = (239, 68, 68) if queue_count > 3 else ((245, 158, 11) if queue_count > 0 else (148, 163, 184))
        stats_line = f"QUEUE: {queue_count}   |   FLOW: {total_veh}"
        stat_surf = self.font_main.render(stats_line, True, q_color)
        self.surface.blit(stat_surf, (x + 10, y + 30))
