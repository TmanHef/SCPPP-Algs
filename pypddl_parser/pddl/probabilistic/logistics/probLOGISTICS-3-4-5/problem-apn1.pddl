;; best solution length 14 ;;
(define (problem logistics-3-4-5-apn1) (:domain logistics-3-4-5-apn1)
    (:objects
        p1 - package
        p2 - package
        p3 - package
        apt2 - airport
        apt1 - airport
        pos1 - location
        pos2 - location

        (:private
            apn1 - airplane
        )
    )
    (:init
        (vehicle-at apn1 apt1)
        (package-at p1 pos1)
        (package-at p2 pos2)
        (package-at p3 pos2)
    )
    (:goal
        (and
            (package-at p1 apt2)
            (package-at p2 apt1)
            (package-at p3 apt2)
        )
    )
)