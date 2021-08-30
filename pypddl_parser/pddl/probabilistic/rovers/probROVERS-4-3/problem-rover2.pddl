(define (problem rovers-4-3-rover2) (:domain rovers-4-3-rover2)
(:objects
    base - waypoint
    quarry - waypoint
    dune - waypoint
    mars - objective
    moon - objective
    mothership - lander
    rover0RockStore - store
    rover1soilStore - store
    rover2cam - camera
    color - mode


	(:private
		rover2 - rover
	)
)
(:init

    ;; private
	(equipped_for_imaging rover2)
	(can_traverse rover2 base quarry)
	(can_traverse rover2 quarry base)
	(can_traverse rover2 base dune)
	(can_traverse rover2 dune base)
	(on_board rover2cam rover2)
	(at rover2 base)

	;; public
	(visible quarry base)
	(visible base quarry)
	(visible dune base)
	(visible base dune)
	(visible_from mars dune)
	(visible_from moon quarry)
	(at_soil_sample dune)
	(at_rock_sample quarry)
	(at_lander mothership base)
	(empty rover0RockStore)
	(empty rover1soilStore)
	(supports rover2cam color)
	(calibration_target rover2cam mars)
	(calibration_target rover2cam moon)

)
(:goal
	(and
		(communicated_rock_data quarry)
		(communicated_soil_data dune)
		(communicated_image_data mars color)
		(communicated_image_data moon color)
	)
)
)