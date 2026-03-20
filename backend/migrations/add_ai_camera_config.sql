-- Migration: Add ai_camera_config table
-- This table stores AI camera configuration for each parking spot

USE `ai_parking`;

CREATE TABLE IF NOT EXISTS `ai_camera_config` (
  `id` int NOT NULL AUTO_INCREMENT,
  `parking_spot_id` int NOT NULL,
  `mode` enum('manual','ai') NOT NULL DEFAULT 'manual',
  `grid_config` json DEFAULT NULL COMMENT 'Grid configuration with cell coordinates',
  `scan_interval` int DEFAULT 30 COMMENT 'Scan interval in seconds',
  `last_calibrated` datetime DEFAULT NULL COMMENT 'Last time grid was calibrated',
  `camera_url` varchar(500) DEFAULT NULL COMMENT 'IP webcam URL',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_parking_spot` (`parking_spot_id`),
  CONSTRAINT `fk_ai_config_parking_spot` FOREIGN KEY (`parking_spot_id`) REFERENCES `parking_spots` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='AI camera configuration per parking spot';

-- Add index for faster lookups
CREATE INDEX `idx_mode` ON `ai_camera_config` (`mode`);
CREATE INDEX `idx_last_calibrated` ON `ai_camera_config` (`last_calibrated`);

-- Optional: Create ai_system_status table if it doesn't exist
CREATE TABLE IF NOT EXISTS `ai_system_status` (
  `id` int NOT NULL AUTO_INCREMENT,
  `parking_spot_id` int NOT NULL,
  `is_running` tinyint(1) DEFAULT 0,
  `fps` decimal(5,2) DEFAULT NULL,
  `last_heartbeat` datetime DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_parking_spot_status` (`parking_spot_id`),
  CONSTRAINT `fk_ai_status_parking_spot` FOREIGN KEY (`parking_spot_id`) REFERENCES `parking_spots` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='AI system runtime status per parking spot';
