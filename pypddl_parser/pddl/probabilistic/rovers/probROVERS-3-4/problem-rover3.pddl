(define (problem rovers-3-4-rover3) (:domain rovers-3-4-rover3)
(:objects
    base - waypoint
    quarry - waypoint
    dune - waypoint
    mars - objective
    mothership - lander
    rover0RockStore - store
    rover1soilStore - store
    rover2RockStore - store
    rover2soilStore - store
    rover3cam - camera
    color - mode


	(:private
		rover3 - rover
	)
)
(:init

    ;; private
	(equipped_for_imaging rover3)
	(can_traverse rover3 base quarry)
	(can_traverse rover3 quarry base)
	(can_traverse rover3 base dune)
	(can_traverse rover3 dune base)
	(on_board rover3cam rover3)
	(at rover3 base)

	;; public
	(visible quarry base)
	(visible base quarry)
	(visible dune base)
	(visible base dune)
	(visible_from mars dune)
	(at_soil_sample dune)
	(at_rock_sample quarry)
	(at_lander mothership base)
	(empty rover0RockStore)
	(empty rover1soilStore)
	(empty rover2RockStore)
	(empty rover2soilStore)
	(supports rover3cam color)
	(calibration_target rover3cam mars)

)
(:goal
	(and
		(communicated_rock_data quarry)
		(communicated_soil_data dune)
		(communicated_image_data mars color)
	)
)
)